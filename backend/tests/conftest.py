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
os.environ.setdefault("EXAM_DATA_DIR", tempfile.mkdtemp(prefix="cat-exam-test-"))
os.environ.setdefault("NVIDIA_API_KEY", "")
