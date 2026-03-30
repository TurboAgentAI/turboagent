"""
Enterprise license validation.

License keys are validated via environment variable or config file.
The core open-source package is fully functional without a license key;
enterprise features are gated behind this check.

Set your license key:
    export TURBOAGENT_LICENSE_KEY="your-key-here"

Or place it in ~/.turboagent/license.key
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("turboagent.enterprise")

_ENTERPRISE_ENV_VAR = "TURBOAGENT_LICENSE_KEY"
_LICENSE_FILE_PATH = Path.home() / ".turboagent" / "license.key"

# Cached validation result (computed once per process)
_cached_result: Optional[bool] = None
_cached_info: Optional[Dict[str, Any]] = None


def _read_license_key() -> Optional[str]:
    """Read the license key from environment or file."""
    # 1. Environment variable (highest priority)
    key = os.environ.get(_ENTERPRISE_ENV_VAR)
    if key:
        return key.strip()

    # 2. License file
    if _LICENSE_FILE_PATH.is_file():
        try:
            return _LICENSE_FILE_PATH.read_text(encoding="utf-8").strip()
        except OSError:
            pass

    return None


def _validate_key(key: str) -> Dict[str, Any]:
    """
    Validate a license key and return license metadata.

    In production this would call a license server or verify a signed
    JWT / cryptographic token. For the MVP scaffold, we accept any
    non-empty key prefixed with "TA-ENT-" as valid.
    """
    if key.startswith("TA-ENT-") and len(key) >= 16:
        return {
            "valid": True,
            "tier": "enterprise",
            "key_prefix": key[:12] + "...",
            "features": [
                "secure_swarm",
                "audit_logging",
                "sso",
                "air_gapped",
                "multi_node_kv",
                "priority_kernels",
            ],
        }

    return {"valid": False, "tier": "community", "key_prefix": None, "features": []}


def is_enterprise_license_valid() -> bool:
    """
    Check if a valid enterprise license is present.

    Returns True if the license key is set and valid.
    Result is cached for the lifetime of the process.
    """
    global _cached_result, _cached_info

    if _cached_result is not None:
        return _cached_result

    key = _read_license_key()
    if key is None:
        _cached_result = False
        _cached_info = {"valid": False, "tier": "community", "key_prefix": None, "features": []}
        return False

    _cached_info = _validate_key(key)
    _cached_result = _cached_info["valid"]

    if _cached_result:
        logger.info(f"Enterprise license validated: {_cached_info['key_prefix']}")
    else:
        logger.warning("Invalid enterprise license key. Running in community mode.")

    return _cached_result


def get_license_info() -> Dict[str, Any]:
    """Return metadata about the current license (tier, features, etc.)."""
    if _cached_info is None:
        is_enterprise_license_valid()  # Populate cache
    return _cached_info or {"valid": False, "tier": "community", "key_prefix": None, "features": []}


def has_feature(feature: str) -> bool:
    """Check if a specific enterprise feature is enabled."""
    info = get_license_info()
    return feature in info.get("features", [])
