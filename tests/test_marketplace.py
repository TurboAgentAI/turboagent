"""Tests for turboagent marketplace: template listing and loading."""

import pytest

from marketplace import list_templates, load_template


class TestListTemplates:
    def test_lists_available_templates(self):
        templates = list_templates()
        assert len(templates) >= 3
        names = [t["name"] for t in templates]
        assert "research_swarm" in names
        assert "code_analyst" in names
        assert "document_qa" in names

    def test_template_has_required_fields(self):
        templates = list_templates()
        for t in templates:
            assert "name" in t
            assert "description" in t
            assert "type" in t


class TestLoadTemplate:
    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_template("nonexistent_template", model="dummy")

    def test_research_swarm_config(self):
        """Verify the research_swarm template parses correctly."""
        import json
        from pathlib import Path

        path = Path(__file__).parent.parent / "marketplace" / "templates" / "research_swarm.json"
        config = json.loads(path.read_text())
        assert config["type"] == "swarm"
        assert len(config["agents"]) == 3
        assert config["agents"][0]["name"] == "researcher"

    def test_code_analyst_config(self):
        import json
        from pathlib import Path

        path = Path(__file__).parent.parent / "marketplace" / "templates" / "code_analyst.json"
        config = json.loads(path.read_text())
        assert config["type"] == "agent"
        assert "system_prompt" in config
