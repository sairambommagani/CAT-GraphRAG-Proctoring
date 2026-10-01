import json

import httpx

from semantic_cache import LangCacheBackend, LocalBackend, SemanticCache


def _cache(**kw):
    return SemanticCache(LocalBackend(None), threshold=0.9, **kw)


def test_paraphrase_hits_and_saves_the_llm_call():
    c, calls = _cache(), []

    def compute():
        calls.append(1)
        return "Overfitting: the model memorises training data and generalises poorly.", 420
    a = c.answer("What is overfitting?", "exam-ask", "v1", compute)
    b = c.answer("What does overfitting mean?", "exam-ask", "v1", compute)
    assert a["cache"]["hit"] is False and b["cache"]["hit"] is True
    assert len(calls) == 1 and b["answer"] == a["answer"]
    assert c.summary()["tokens_saved"] == 420 and c.summary()["hit_rate"] == 0.5


def test_different_question_misses():
    c = _cache()
    c.answer("What is overfitting?", "exam-ask", "v1", lambda: ("x", 10))
    r = c.answer("Which section has the fewest approved questions?", "exam-ask", "v1", lambda: ("y", 10))
    assert r["cache"]["hit"] is False and r["answer"] == "y"


def test_scopes_are_separate_and_new_knowledge_purges_old_answers():
    c = _cache()
    c.answer("What is overfitting?", "exam-ask", "v1", lambda: ("old", 10))
    assert c.answer("What is overfitting?", "proctor-ask", "v1", lambda: ("p", 10))["cache"]["hit"] is False
    r = c.answer("What is overfitting?", "exam-ask", "v2", lambda: ("new", 10))     # graph changed
    assert r["cache"]["hit"] is False and r["answer"] == "new"
    assert all(e["attributes"]["kb"] == "v2" for e in c.backend.entries if e["attributes"]["scope"] == "exam-ask")


def test_cache_failure_never_blocks_the_answer():
    class Broken:
        name = "broken"

        def search(self, *a):
            raise RuntimeError("down")

        def set(self, *a):
            raise RuntimeError("down")

        def purge(self, *a):
            raise RuntimeError("down")
    c = SemanticCache(Broken())
    r = c.answer("q", "exam-ask", "v1", lambda: ("fine", 5))
    assert r["answer"] == "fine" and c.summary()["errors"] == 2


def test_embedding_backend_used_and_degrades_to_word_overlap():
    vecs = {"What is overfitting?": [1.0, 0.0], "Explain overfitting to me": [0.95, 0.05]}
    b = LocalBackend(lambda t: vecs.get(t, [0.0, 1.0]))
    c = SemanticCache(b, threshold=0.9)
    c.answer("What is overfitting?", "s", "v", lambda: ("A", 1))
    assert c.answer("Explain overfitting to me", "s", "v", lambda: ("B", 1))["answer"] == "A"

    def boom(t):
        raise httpx.ConnectError("no network")
    b2 = LocalBackend(boom)
    c2 = SemanticCache(b2)
    c2.answer("What is overfitting?", "s", "v", lambda: ("A", 1))
    assert "word overlap" in c2.backend_name


def test_langcache_rest_calls():
    seen = []

    def handler(req: httpx.Request):
        body = json.loads(req.content or b"{}")
        seen.append((req.method, req.url.path, body, req.headers.get("authorization")))
        if req.url.path.endswith("/search"):
            return httpx.Response(200, json={"data": [{"id": "e1", "prompt": "What is overfitting?",
                                                       "response": "cached", "similarity": 0.95}]})
        return httpx.Response(201, json={"entryId": "e2"})
    be = LangCacheBackend("abc.redis.io", "cache1", "key1", client=httpx.Client(transport=httpx.MockTransport(handler)))
    c = SemanticCache(be, threshold=0.9)
    r = c.answer("What does overfitting mean?", "exam-ask", "v1", lambda: ("never", 1))
    assert r["cache"]["hit"] and r["answer"] == "cached" and r["cache"]["similarity"] == 0.95
    m, path, body, auth = seen[0]
    assert m == "POST" and path == "/v1/caches/cache1/entries/search" and auth == "Bearer key1"
    assert body["attributes"] == {"scope": "exam-ask", "kb": "v1"} and body["similarityThreshold"] == 0.9


def test_langcache_below_threshold_is_a_miss_and_stores():
    seen = []

    def handler(req):
        seen.append(req.url.path)
        if req.url.path.endswith("/search"):
            return httpx.Response(200, json={"data": [{"response": "x", "similarity": 0.5}]})
        return httpx.Response(201, json={})
    be = LangCacheBackend("https://h", "c", "k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    r = SemanticCache(be).answer("q", "s", "v", lambda: ("fresh", 3))
    assert r["answer"] == "fresh" and r["cache"]["stored"] and seen[-1] == "/v1/caches/c/entries"


def test_exam_ask_goes_through_the_cache(tmp_path, monkeypatch):
    import semantic_cache
    from exam_graph.service import ExamService
    monkeypatch.setattr(semantic_cache, "_shared", SemanticCache(LocalBackend(None)))

    class FakeLLM:
        last_model, last_tokens, calls = "nvidia/fake", 300, 0

        def chat(self, system, user, **kw):
            FakeLLM.calls += 1
            return "Model Evaluation is weakest."

        def chat_json(self, *a, **k):
            raise RuntimeError("not used")
    svc = ExamService(tmp_path, llm=FakeLLM())
    a = svc.ask("Which section do candidates find hardest?")
    b = svc.ask("Which section do candidates find most hard?")
    assert a["cache"]["hit"] is False and b["cache"]["hit"] is True and FakeLLM.calls == 1
    assert b["answer"] == "Model Evaluation is weakest." and "graph_answer" in b


def test_langcache_without_configured_attributes_falls_back_to_prompt_prefix():
    seen = []

    def handler(req):
        body = json.loads(req.content or b"{}")
        seen.append(body)
        if "attributes" in body:
            return httpx.Response(400, json={"detail": "attribute 'scope' is not configured"})
        if req.url.path.endswith("/search"):
            hit = [{"prompt": "[exam-ask|v1] What is overfitting?", "response": "cached", "similarity": 0.97}]
            return httpx.Response(200, json={"data": hit if len(seen) > 3 else []})
        return httpx.Response(201, json={})
    be = LangCacheBackend("https://h", "c", "k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    c = SemanticCache(be)
    r1 = c.answer("What is overfitting?", "exam-ask", "v1", lambda: ("fresh", 5))
    assert r1["answer"] == "fresh" and r1["cache"]["stored"] and not be.use_attributes
    assert seen[-1]["prompt"].startswith("[exam-ask|v1] ")
    r2 = c.answer("What does overfitting mean?", "exam-ask", "v1", lambda: ("never", 5))
    assert r2["cache"]["hit"] and r2["cache"]["matched_question"] == "What is overfitting?"
