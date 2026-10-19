import os
import sys

# Forward imports to backend.router to ensure absolute consistency
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from backend.router import (
    DOMAIN_KEYWORDS,
    OUT_OF_SCOPE_RESPONSE,
    IntentRouter,
    is_astrology_query,
    validate_astrology_domain,
)

__all__ = [
    "DOMAIN_KEYWORDS",
    "OUT_OF_SCOPE_RESPONSE",
    "IntentRouter",
    "is_astrology_query",
    "validate_astrology_domain",
]