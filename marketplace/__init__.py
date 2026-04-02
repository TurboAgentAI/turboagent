"""
TurboAgent Marketplace — pre-built agent templates.

Load and instantiate pre-configured agent pipelines from template definitions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger("turboagent.marketplace")

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def list_templates() -> list[dict]:
    """List all available marketplace templates."""
    templates = []
    for f in sorted(_TEMPLATES_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            templates.append({
                "name": data.get("name", f.stem),
                "description": data.get("description", ""),
                "type": data.get("type", "agent"),
                "file": f.name,
            })
        except Exception as e:
            logger.warning(f"Failed to load template {f.name}: {e}")
    return templates


def load_template(
    name: str,
    model: str,
    **overrides: Any,
) -> Any:
    """
    Load and instantiate a marketplace template.

    Args:
        name: Template name (e.g., "research_swarm").
        model: Model ID to use.
        **overrides: Override any template config values.

    Returns:
        A configured TurboAgent or TurboSwarm instance.
    """
    template_path = _TEMPLATES_DIR / f"{name}.json"
    if not template_path.exists():
        available = [f.stem for f in _TEMPLATES_DIR.glob("*.json")]
        raise FileNotFoundError(
            f"Template '{name}' not found. Available: {available}"
        )

    config = json.loads(template_path.read_text(encoding="utf-8"))
    config.update(overrides)
    template_type = config.get("type", "agent")

    if template_type == "swarm":
        return _build_swarm(model, config)
    else:
        return _build_agent(model, config)


def _build_agent(model: str, config: dict) -> Any:
    """Build a TurboAgent from template config."""
    from turboagent.agents.base import TurboAgent

    return TurboAgent(
        model,
        system_prompt=config.get("system_prompt"),
        kv_mode=config.get("kv_mode", "turbo3"),
        context=config.get("context", 131072),
    )


def _build_swarm(model: str, config: dict) -> Any:
    """Build a TurboSwarm from template config."""
    from turboagent.agents.swarm import TurboSwarm, SwarmAgent

    agents = []
    for agent_cfg in config.get("agents", []):
        agents.append(SwarmAgent(
            name=agent_cfg["name"],
            role=agent_cfg["role"],
            system_prompt=agent_cfg.get("system_prompt"),
        ))

    return TurboSwarm(
        model,
        agents=agents,
        max_rounds=config.get("max_rounds", 10),
        kv_mode=config.get("kv_mode", "turbo3"),
        context=config.get("context", 131072),
    )
