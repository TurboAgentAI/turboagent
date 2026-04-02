"""
TurboAgent Enterprise Edition.

Commercial extensions for teams and organizations requiring:
  - SSO / SAML authentication
  - Audit logging and compliance exports (SOC-2, GDPR)
  - Air-gapped / on-premise licensing
  - Advanced swarm governance and role-based access control
  - Multi-node KV cache sharing
  - Priority kernel optimizations
  - Dedicated support SLAs

License: Commercial (separate from MIT core).
Contact: enterprise@turboagent.to
More info: https://turboagent.to/enterprise
"""

from turboagent.enterprise.license import is_enterprise_license_valid, get_license_info

__all__ = ["is_enterprise_license_valid", "get_license_info"]
