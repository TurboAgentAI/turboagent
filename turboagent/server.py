"""
TurboAgent Cloud API Server — OpenAI-compatible /v1/chat/completions.

Wraps TurboAgent with a FastAPI server that provides:
  - OpenAI-compatible chat completions endpoint
  - Persistent, TurboQuant-compressed sessions per API key
  - API key authentication with configurable rate limiting
  - Health check and model info endpoints

Usage:
    turboagent serve --model meta-llama/Llama-3.1-70B-Instruct --port 8000
    # or
    uvicorn turboagent.server:create_app --factory --host 0.0.0.0 --port 8000

Environment variables:
    TURBOAGENT_MODEL       Model ID (HF hub or local path)
    TURBOAGENT_BACKEND     Backend: llama.cpp, torch, vllm (default: auto)
    TURBOAGENT_KV_MODE     turbo3 or turbo4 (default: turbo3)
    TURBOAGENT_CONTEXT     Max context length (default: auto)
    TURBOAGENT_API_KEYS    Comma-separated valid API keys (empty = no auth)
    TURBOAGENT_RATE_LIMIT  Requests per minute per key (default: 60)

This is part of the open-source TurboAgent core (MIT license).
TurboAgent Cloud (managed hosting) available at https://turboagent.to/cloud
"""

import logging
import os
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from turboagent.version import __version__

logger = logging.getLogger("turboagent.server")


# ---------------------------------------------------------------------------
# Request / Response models (OpenAI-compatible)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "turboagent"
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "turboagent"


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = __version__
    model: str = ""
    backend: str = ""
    kv_mode: str = ""
    kv_compressed_mb: float = 0.0


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple in-memory sliding-window rate limiter per API key."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def check(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True


# ---------------------------------------------------------------------------
# Session manager (per-key persistent KV cache)
# ---------------------------------------------------------------------------

class SessionManager:
    """Manages per-session TurboAgent instances with compressed KV cache."""

    def __init__(self, model_id: str, **engine_kwargs):
        self.model_id = model_id
        self.engine_kwargs = engine_kwargs
        self._sessions: Dict[str, Any] = {}

    def get_or_create(self, session_id: str):
        if session_id not in self._sessions:
            from turboagent.backends import create_engine
            from turboagent.quant.turboquant import TurboQuantKVCache
            from turboagent.hardware.detector import HardwareDetector

            config = HardwareDetector.get_optimal_config()
            config.update(self.engine_kwargs)

            engine = create_engine(self.model_id, **config)

            flat_dim = getattr(engine, "_head_dim", 128) * getattr(engine, "_n_kv_heads", 8)
            n_layers = getattr(engine, "_n_layers", 80)

            cache = TurboQuantKVCache(
                bit_mode=config.get("kv_mode", "turbo3"),
                device="cpu",
                head_dim=flat_dim,
                num_layers=n_layers,
                max_context=config.get("context", 131072),
            )

            self._sessions[session_id] = {
                "engine": engine,
                "cache": cache,
                "config": config,
                "history": [],
                "created": time.time(),
            }
            logger.info(f"Created session {session_id}")

        return self._sessions[session_id]

    def remove(self, session_id: str):
        self._sessions.pop(session_id, None)

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    model_id: Optional[str] = None,
    backend: Optional[str] = None,
    kv_mode: str = "turbo3",
    context: Optional[int] = None,
    api_keys: Optional[List[str]] = None,
    rate_limit: int = 60,
) -> FastAPI:
    """
    Create the FastAPI application.

    Can be called directly or via uvicorn factory:
        uvicorn turboagent.server:create_app --factory
    """
    # Resolve config from args or environment
    model_id = model_id or os.environ.get("TURBOAGENT_MODEL", "")
    backend = backend or os.environ.get("TURBOAGENT_BACKEND", "")
    kv_mode = os.environ.get("TURBOAGENT_KV_MODE", kv_mode)
    rate_limit = int(os.environ.get("TURBOAGENT_RATE_LIMIT", str(rate_limit)))

    if context is None:
        ctx_env = os.environ.get("TURBOAGENT_CONTEXT", "")
        context = int(ctx_env) if ctx_env else None

    if api_keys is None:
        keys_env = os.environ.get("TURBOAGENT_API_KEYS", "")
        api_keys = [k.strip() for k in keys_env.split(",") if k.strip()] if keys_env else []

    # Build engine kwargs
    engine_kwargs: Dict[str, Any] = {"kv_mode": kv_mode}
    if backend:
        engine_kwargs["backend"] = backend
    if context:
        engine_kwargs["context"] = context

    # State (initialized in lifespan)
    rate_limiter = RateLimiter(max_requests=rate_limit)
    session_mgr: Optional[SessionManager] = None
    require_auth = len(api_keys) > 0
    valid_keys = set(api_keys)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal session_mgr
        if model_id:
            session_mgr = SessionManager(model_id, **engine_kwargs)
            logger.info(f"TurboAgent server ready | Model: {model_id} | KV: {kv_mode}")
        else:
            logger.warning(
                "No model specified. Set TURBOAGENT_MODEL or pass --model to turboagent serve."
            )
        yield
        # Shutdown: cleanup sessions
        logger.info("Shutting down TurboAgent server.")

    app = FastAPI(
        title="TurboAgent API",
        description="OpenAI-compatible API with TurboQuant KV cache compression",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Auth middleware ---

    @app.middleware("http")
    async def auth_and_rate_limit(request: Request, call_next):
        # Skip auth for health/docs
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # API key check
        if require_auth:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] not in valid_keys:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Invalid API key", "type": "auth_error"}},
                )
            key = auth[7:]
        else:
            key = "anonymous"

        # Rate limit
        if not rate_limiter.check(key):
            return JSONResponse(
                status_code=429,
                content={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
            )

        return await call_next(request)

    # --- Endpoints ---

    @app.get("/health", response_model=HealthResponse)
    async def health():
        resp = HealthResponse(
            version=__version__,
            model=model_id or "not loaded",
            backend=backend or "auto",
            kv_mode=kv_mode,
        )
        if session_mgr and session_mgr.active_sessions > 0:
            # Report aggregate KV compression from first session
            first = next(iter(session_mgr._sessions.values()), None)
            if first:
                resp.kv_compressed_mb = first["cache"].memory_usage_gb() * 1000
        return resp

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": model_id or "turboagent", "object": "model", "owned_by": "turboagent"}],
        }

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
        if request.stream:
            raise HTTPException(status_code=501, detail="Streaming not yet supported.")

        if session_mgr is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Set TURBOAGENT_MODEL.")

        # Session ID from header or generate one
        session_id = raw_request.headers.get("X-Session-ID", "default")

        try:
            session = session_mgr.get_or_create(session_id)
            engine = session["engine"]
            cache = session["cache"]

            messages = [{"role": m.role, "content": m.content} for m in request.messages]

            t0 = time.time()
            response_text, metrics = engine.generate_chat(
                messages=messages,
                kv_cache=cache,
            )
            latency = time.time() - t0

            logger.info(
                f"Session {session_id} | "
                f"In: {metrics.get('turn_input_tokens', 0)} | "
                f"Out: {metrics.get('turn_output_tokens', 0)} | "
                f"Latency: {latency:.2f}s | "
                f"KV: {cache.memory_usage_gb()*1000:.1f} MB"
            )

            return ChatCompletionResponse(
                model=model_id or "turboagent",
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessage(role="assistant", content=response_text),
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=metrics.get("turn_input_tokens", 0),
                    completion_tokens=metrics.get("turn_output_tokens", 0),
                    total_tokens=metrics.get("total_tokens_cached", 0),
                ),
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str):
        if session_mgr:
            session_mgr.remove(session_id)
        return {"status": "deleted", "session_id": session_id}

    return app
