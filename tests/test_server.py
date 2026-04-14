"""
Tests for turboagent.server: FastAPI endpoints, auth, rate limiting.

Uses FastAPI's TestClient (no real model needed for most tests).
"""

import pytest

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestServerHealth:
    @pytest.fixture
    def client(self):
        from turboagent.server import create_app
        app = create_app(model_id="test-model")
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert data["model"] == "test-model"

    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestServerAuth:
    @pytest.fixture
    def client_with_auth(self):
        from turboagent.server import create_app
        app = create_app(model_id="test-model", api_keys=["sk-testkey1", "sk-testkey2"])
        return TestClient(app)

    def test_no_key_rejected(self, client_with_auth):
        resp = client_with_auth.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 401

    def test_wrong_key_rejected(self, client_with_auth):
        resp = client_with_auth.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk-wrongkey"},
        )
        assert resp.status_code == 401

    def test_valid_key_accepted(self, client_with_auth):
        # Will fail at generation (no real model) but should pass auth
        resp = client_with_auth.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk-testkey1"},
        )
        # 500 (model not real) or 503, not 401
        assert resp.status_code != 401

    def test_health_no_auth_needed(self, client_with_auth):
        resp = client_with_auth.get("/health")
        assert resp.status_code == 200


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestServerNoAuth:
    @pytest.fixture
    def client_open(self):
        from turboagent.server import create_app
        app = create_app(model_id="test-model", api_keys=[])
        return TestClient(app)

    def test_no_auth_passes(self, client_open):
        # No auth required — should reach the model layer (and fail there)
        resp = client_open.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code != 401


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestRateLimiting:
    def test_rate_limit_enforced(self):
        from turboagent.server import create_app
        app = create_app(model_id="test-model", api_keys=["sk-key"], rate_limit=3)
        client = TestClient(app)
        headers = {"Authorization": "Bearer sk-key"}

        # First 3 should pass (or fail at model layer, not at rate limit)
        for _ in range(3):
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers=headers,
            )
            assert resp.status_code != 429

        # 4th should be rate limited
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=headers,
        )
        assert resp.status_code == 429


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestSessionManagement:
    @pytest.fixture
    def client(self):
        from turboagent.server import create_app
        app = create_app(model_id="test-model")
        return TestClient(app)

    def test_delete_session(self, client):
        resp = client.delete("/v1/sessions/test-session")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_stream_not_supported(self, client):
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 501
