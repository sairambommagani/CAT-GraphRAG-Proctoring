import os
import tempfile

# Test defaults (set before app.main is imported):
#  - the original CAT API tests exercise the IRT engine without a webcam
#  - evidence goes to a throwaway directory; the offline judge avoids API calls
os.environ.setdefault("CAT_REQUIRE_PROCTORING", "0")
os.environ.setdefault("PROCTOR_DATA_DIR", tempfile.mkdtemp(prefix="cat-proctor-test-"))
os.environ.setdefault("PROCTOR_MOCK_JUDGE", "1")
os.environ.setdefault("PROCTOR_ADMIN_TOKEN", "test-admin")
os.environ.setdefault("PROCTOR_ASR", "0")
os.environ.setdefault("CAT_REQUIRE_LOGIN", "0")
os.environ.setdefault("CAT_ATTEMPTS_PATH", os.path.join(__import__("tempfile").mkdtemp(), "attempts.json"))
os.environ.setdefault("CAT_ACCOUNTS_PATH", os.path.join(__import__("tempfile").mkdtemp(), "accounts.json"))
os.environ.setdefault("EXAM_DATA_DIR", tempfile.mkdtemp(prefix="cat-exam-test-"))
os.environ.setdefault("NVIDIA_API_KEY", "")


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_semantic_cache():
    """Each test gets its own (local, in-memory) semantic cache."""
    import semantic_cache
    semantic_cache._shared = None
    yield
    semantic_cache._shared = None
