"""
Tests for turboagent.enterprise: license validation, feature flags, audit logging.
"""

import os

import pytest


class TestLicenseValidation:
    def setup_method(self):
        """Reset cached license state between tests."""
        from turboagent.enterprise import license as lic_mod

        lic_mod._cached_result = None
        lic_mod._cached_info = None

    def teardown_method(self):
        """Clean up env var after each test."""
        os.environ.pop("TURBOAGENT_LICENSE_KEY", None)
        from turboagent.enterprise import license as lic_mod

        lic_mod._cached_result = None
        lic_mod._cached_info = None

    def test_no_key_returns_false(self):
        from turboagent.enterprise.license import is_enterprise_license_valid

        os.environ.pop("TURBOAGENT_LICENSE_KEY", None)
        assert is_enterprise_license_valid() is False

    def test_invalid_key_returns_false(self):
        from turboagent.enterprise.license import is_enterprise_license_valid

        os.environ["TURBOAGENT_LICENSE_KEY"] = "INVALID-KEY"
        assert is_enterprise_license_valid() is False

    def test_short_key_returns_false(self):
        from turboagent.enterprise.license import is_enterprise_license_valid

        os.environ["TURBOAGENT_LICENSE_KEY"] = "TA-ENT-short"
        assert is_enterprise_license_valid() is False

    def test_valid_key_returns_true(self):
        from turboagent.enterprise.license import is_enterprise_license_valid

        os.environ["TURBOAGENT_LICENSE_KEY"] = "TA-ENT-0123456789abcdef"
        assert is_enterprise_license_valid() is True

    def test_get_license_info_community(self):
        from turboagent.enterprise.license import get_license_info

        os.environ.pop("TURBOAGENT_LICENSE_KEY", None)
        info = get_license_info()
        assert info["tier"] == "community"
        assert info["valid"] is False
        assert info["features"] == []

    def test_get_license_info_enterprise(self):
        from turboagent.enterprise.license import get_license_info

        os.environ["TURBOAGENT_LICENSE_KEY"] = "TA-ENT-0123456789abcdef"
        info = get_license_info()
        assert info["tier"] == "enterprise"
        assert info["valid"] is True
        assert "secure_swarm" in info["features"]
        assert "audit_logging" in info["features"]

    def test_has_feature_with_valid_key(self):
        from turboagent.enterprise.license import has_feature

        os.environ["TURBOAGENT_LICENSE_KEY"] = "TA-ENT-0123456789abcdef"
        assert has_feature("secure_swarm") is True
        assert has_feature("nonexistent_feature") is False

    def test_has_feature_without_key(self):
        from turboagent.enterprise.license import has_feature

        os.environ.pop("TURBOAGENT_LICENSE_KEY", None)
        assert has_feature("secure_swarm") is False

    def test_result_is_cached(self):
        from turboagent.enterprise.license import is_enterprise_license_valid
        from turboagent.enterprise import license as lic_mod

        os.environ["TURBOAGENT_LICENSE_KEY"] = "TA-ENT-0123456789abcdef"
        assert is_enterprise_license_valid() is True

        # Change env var — cached result should persist
        os.environ["TURBOAGENT_LICENSE_KEY"] = "INVALID"
        assert is_enterprise_license_valid() is True  # Still cached as True


class TestSecureSwarm:
    def setup_method(self):
        from turboagent.enterprise import license as lic_mod

        lic_mod._cached_result = None
        lic_mod._cached_info = None

    def teardown_method(self):
        os.environ.pop("TURBOAGENT_LICENSE_KEY", None)
        from turboagent.enterprise import license as lic_mod

        lic_mod._cached_result = None
        lic_mod._cached_info = None

    def test_secure_swarm_requires_license(self):
        from turboagent.enterprise.swarm import SecureMultiAgentSwarm
        from turboagent.agents.swarm import SwarmAgent

        os.environ.pop("TURBOAGENT_LICENSE_KEY", None)
        with pytest.raises(RuntimeError, match="enterprise license"):
            SecureMultiAgentSwarm(
                model_id="dummy",
                agents=[SwarmAgent(name="a", role="test")],
            )


class TestAuditLogger:
    def setup_method(self):
        from turboagent.enterprise import license as lic_mod

        lic_mod._cached_result = None
        lic_mod._cached_info = None

    def teardown_method(self):
        os.environ.pop("TURBOAGENT_LICENSE_KEY", None)
        from turboagent.enterprise import license as lic_mod

        lic_mod._cached_result = None
        lic_mod._cached_info = None

    def test_audit_logger_requires_license(self):
        from turboagent.enterprise.audit import AuditLogger

        os.environ.pop("TURBOAGENT_LICENSE_KEY", None)
        with pytest.raises(RuntimeError, match="enterprise license"):
            AuditLogger()

    def test_audit_logger_with_license(self):
        from turboagent.enterprise.audit import AuditLogger

        os.environ["TURBOAGENT_LICENSE_KEY"] = "TA-ENT-0123456789abcdef"
        logger = AuditLogger()
        logger.log(actor="test", action="test_action", detail="hello")
        assert len(logger) == 1

    def test_audit_export_json(self):
        import json

        from turboagent.enterprise.audit import AuditLogger

        os.environ["TURBOAGENT_LICENSE_KEY"] = "TA-ENT-0123456789abcdef"
        logger = AuditLogger()
        logger.log(actor="agent", action="run", detail="prompt")
        export = json.loads(logger.export(format="json"))
        assert len(export) == 1
        assert export[0]["actor"] == "agent"

    def test_audit_export_csv(self):
        from turboagent.enterprise.audit import AuditLogger

        os.environ["TURBOAGENT_LICENSE_KEY"] = "TA-ENT-0123456789abcdef"
        logger = AuditLogger()
        logger.log(actor="agent", action="run", detail="hello")
        csv = logger.export(format="csv")
        lines = csv.strip().split("\n")
        assert lines[0] == "timestamp,actor,action,detail"
        assert "agent" in lines[1]

    def test_audit_clear(self):
        from turboagent.enterprise.audit import AuditLogger

        os.environ["TURBOAGENT_LICENSE_KEY"] = "TA-ENT-0123456789abcdef"
        logger = AuditLogger()
        logger.log(actor="a", action="b")
        logger.clear()
        assert len(logger) == 0
