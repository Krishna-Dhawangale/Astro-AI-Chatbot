"""
Frontend XSS Prevention Verification Runner (scratch/test_frontend_xss_security.py)
-----------------------------------------------------------------------------------
Verifies:
1. Static code analysis: frontend/script.js has 0 occurrences of .innerHTML property access.
2. Safe DOM rendering methods (document.createTextNode, textContent, appendChild) are used.
3. Automated execution of Node.js security test suite (tests/test_xss_prevention.js).
"""

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_JS = PROJECT_ROOT / "frontend" / "script.js"
TEST_JS = PROJECT_ROOT / "tests" / "test_xss_prevention.js"


def test_static_script_hygiene():
    print("[Check 1] Verifying frontend/script.js static hygiene...")
    assert SCRIPT_JS.exists(), f"Missing {SCRIPT_JS}"
    content = SCRIPT_JS.read_text(encoding="utf-8")

    # Ensure zero .innerHTML property access
    has_inner_html = re.search(r"\.innerHTML\b", content)
    assert not has_inner_html, f"SECURITY VIOLATION: .innerHTML property access found: {has_inner_html.group()}"
    print("  [OK] Zero .innerHTML property access found in frontend/script.js")

    # Ensure DOM safe methods are present
    assert "createTextNode" in content, "Missing document.createTextNode in frontend/script.js"
    assert "textContent" in content, "Missing textContent in frontend/script.js"
    assert "createElement" in content, "Missing document.createElement in frontend/script.js"
    print("  [OK] Safe DOM manipulation primitives confirmed (createTextNode, textContent, createElement)")


def test_node_xss_suite():
    print("\n[Check 2] Executing Node.js XSS Prevention Test Suite...")
    assert TEST_JS.exists(), f"Missing {TEST_JS}"
    result = subprocess.run(
        ["node", str(TEST_JS)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, f"Node XSS test suite failed with code {result.returncode}"
    assert "TOTAL TESTS: 20 | PASSED: 20 | FAILED: 0" in result.stdout
    print("  [OK] All 20 frontend XSS and streaming security tests passed successfully!")


if __name__ == "__main__":
    print("=" * 66)
    print("   RUNNING FRONTEND SECURITY & XSS VERIFICATION (STEP 2)")
    print("=" * 66)
    test_static_script_hygiene()
    test_node_xss_suite()
    print("\n" + "=" * 66)
    print("   STEP 2 VERIFICATION COMPLETE: ALL CHECKS PASSED (100%)")
    print("=" * 66)
