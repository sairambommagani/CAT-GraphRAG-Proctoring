"""Semantic cache for LLM answers (Redis LangCache, with a local fallback).

Why: examiners ask the same things in different words ("Which topics do candidates
fail most?" / "What are students weakest in?"). A normal cache needs the exact same
text; a semantic cache compares the MEANING (embeddings + similarity threshold) and
returns the stored answer - no LLM call, no tokens, milliseconds instead of seconds.

Backends (picked automatically):
  1. Redis LangCache (managed, Redis Cloud) when LANGCACHE_SERVER_URL, LANGCACHE_CACHE_ID
     and LANGCACHE_API_KEY are set. REST API: POST /v1/caches/{id}/entries/search, POST
     /v1/caches/{id}/entries, DELETE /v1/caches/{id}/entries (by attributes).
  2. Local semantic cache otherwise: NVIDIA NIM embeddings (same NVIDIA key) + cosine
     similarity, kept in memory; falls back to word overlap if embeddings are unavailable.

What is cached, and what is NOT:
  * cached   - examiner questions answered by the LLM over the knowledge graphs.
  * never    - AI-judge verdicts (every candidate's clip is judged on its own; reusing a
               "similar" verdict would be unfair) and question generation (it must produce
               new questions). Those keep their exact-match handling.
Entries carry attributes {scope, kb}: `kb` is a version of the graph the answer came from,
so when the data changes (new incidents, erasure, new questions) old answers stop matching
and the scope is purged.
"""
from __future__ import annotations

import math
import os
import re
import threading
import time
from collections import Counter
from typing import Callable, Optional

import httpx

_STOP = set("a an the is are was were be been do does did what which who whom whose how why when where of in on "
            "for to by with about from at as and or me my our we you your i it its this that these those there "
            "most more much many any some all can could should would will shall may might tell show list give "
            "explain describe mean means meaning please".split())


def _words(text: str) -> list[str]:
    return [w.rstrip("s") if len(w) > 4 else w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


def _cos(a, b) -> float:
    if isinstance(a, Counter):
        num = sum(a[k] * b.get(k, 0) for k in a)
        den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    else:
        num = sum(x * y for x, y in zip(a, b))
        den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0


class LangCacheBackend:
    name = "Redis LangCache"

    def __init__(self, server_url: str, cache_id: str, api_key: str, client: Optional[httpx.Client] = None):
        url = server_url.strip().rstrip("/")
        self.base = (url if url.startswith("http") else f"https://{url}") + f"/v1/caches/{cache_id.strip()}/entries"
        self.headers = {"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"}
        self.client = client or httpx.Client(timeout=10)

    use_attributes = True          # turned off if this cache has no attributes configured (HTTP 400)

    def _post(self, path: str, body: dict) -> httpx.Response:
        r = self.client.post(self.base + path, headers=self.headers, json=body)
        if r.status_code == 400 and "attributes" in body and self.use_attributes:
            # LangCache only accepts attributes that were defined when the cache was created;
            # without them, fall back to scoping by a prompt prefix
            self.use_attributes = False
            print(f"[cache] LangCache rejected attributes ({r.text[:160]}); using prompt-prefix scoping", flush=True)
            return self._post(path, self._scoped(body))
        if r.status_code >= 400:
            raise RuntimeError(f"LangCache HTTP {r.status_code}: {r.text[:300]}")
        return r

    def _scoped(self, body: dict) -> dict:
        body = dict(body)
        attrs = body.pop("attributes", None) or {}
        if attrs:
            body["prompt"] = f"[{attrs.get('scope', '')}|{attrs.get('kb', '')}] {body['prompt']}"
        return body

    def search(self, prompt: str, attributes: dict, threshold: float) -> Optional[dict]:
        body = {"prompt": prompt, "attributes": attributes, "similarityThreshold": threshold}
        r = self._post("/search", body if self.use_attributes else self._scoped(body))
        body = r.json() if r.content else {}
        items = body.get("data", body) if isinstance(body, dict) else body
        if isinstance(items, dict):
            items = [items]
        best = None
        for it in items or []:
            sim = float(it.get("similarity", it.get("score", 0)) or 0)
            if sim >= threshold and (best is None or sim > best["similarity"]):
                best = {"response": it.get("response", ""), "similarity": sim,
                        "prompt": re.sub(r"^\[[^\]]*\] ", "", it.get("prompt", "") or ""),
                        "id": it.get("id") or it.get("entryId")}
        return best

    def set(self, prompt: str, response: str, attributes: dict, tokens: int) -> None:
        body = {"prompt": prompt, "response": response, "attributes": attributes}
        self._post("", body if self.use_attributes else self._scoped(body))

    def purge(self, attributes: dict) -> None:
        if not self.use_attributes:
            return                     # entries are keyed by version in the prompt prefix; old ones never match
        r = self.client.request("DELETE", self.base, headers=self.headers, json={"attributes": attributes})
        if r.status_code not in (200, 204, 404):
            raise RuntimeError(f"LangCache HTTP {r.status_code}: {r.text[:300]}")


class LocalBackend:
    """In-process semantic cache: NVIDIA NIM embeddings + cosine (word overlap as a fallback)."""

    def __init__(self, embed: Optional[Callable[[str], Optional[list]]] = None, ttl_s: float = 6 * 3600,
                 max_entries: int = 500):
        self.embed = embed
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.entries: list[dict] = []
        self.lock = threading.Lock()
        self.mode = "NIM embeddings" if embed else "word overlap"

    @property
    def name(self) -> str:
        return f"local semantic cache ({self.mode})"

    def _vec(self, text: str):
        if self.embed is not None:
            try:
                v = self.embed(text)
                if v:
                    return ("emb", v)
            except Exception:
                pass
            self.mode = "word overlap"                      # embeddings unavailable: degrade, keep working
            self.embed = None
        return ("bow", Counter(_words(text)))

    def search(self, prompt: str, attributes: dict, threshold: float) -> Optional[dict]:
        kind, v = self._vec(prompt)
        now, best = time.time(), None
        with self.lock:
            self.entries = [e for e in self.entries if now - e["t"] < self.ttl_s]
            for e in self.entries:
                if e["attributes"] != attributes or e["kind"] != kind:
                    continue
                # word overlap is cruder than embeddings, so it needs a lower bar to match paraphrases
                sim = _cos(v, e["vec"])
                if sim >= (threshold if kind == "emb" else min(threshold, 0.7)) and (best is None or sim > best["similarity"]):
                    best = {"response": e["response"], "similarity": sim, "prompt": e["prompt"], "tokens": e["tokens"]}
        return best

    def set(self, prompt: str, response: str, attributes: dict, tokens: int) -> None:
        kind, v = self._vec(prompt)
        with self.lock:
            self.entries.append({"prompt": prompt, "response": response, "attributes": dict(attributes),
                                 "kind": kind, "vec": v, "tokens": tokens, "t": time.time()})
            del self.entries[:-self.max_entries]

    def purge(self, attributes: dict) -> None:
        with self.lock:
            self.entries = [e for e in self.entries
                            if any(e["attributes"].get(k) != v for k, v in attributes.items())]


def nim_embedder(api_key: str, model: str, base_url: str) -> Callable[[str], Optional[list]]:
    client = httpx.Client(timeout=15)

    def embed(text: str):
        r = client.post(f"{base_url.rstrip('/')}/embeddings", headers={"Authorization": f"Bearer {api_key}"},
                        json={"model": model, "input": [text[:2000]], "input_type": "query", "encoding_format": "float",
                              "truncate": "END"})
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]
    return embed


class SemanticCache:
    def __init__(self, backend, threshold: float = 0.9):
        self.backend = backend
        self.threshold = threshold
        self.lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "tokens_saved": 0, "llm_seconds_saved": 0.0, "errors": 0}
        self._kb_seen: dict[str, str] = {}
        self.last_error: Optional[str] = None

    @classmethod
    def from_env(cls) -> "SemanticCache":
        env = os.environ.get
        thr = float(env("SEMANTIC_CACHE_THRESHOLD", "0.9"))
        url, cid, key = env("LANGCACHE_SERVER_URL", ""), env("LANGCACHE_CACHE_ID", ""), env("LANGCACHE_API_KEY", "")
        if url and cid and key:
            return cls(LangCacheBackend(url, cid, key), thr)
        nv = env("NVIDIA_API_KEY") or env("NIM_API_KEY")
        embed = None
        if nv and not nv.startswith("nvapi-xxx") and env("SEMANTIC_CACHE_EMBEDDINGS", "1") != "0":
            embed = nim_embedder(nv, env("SEMANTIC_CACHE_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5"),
                                 env("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"))
        return cls(LocalBackend(embed), thr)

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def _note_kb(self, scope: str, kb: str) -> None:
        """Knowledge changed since the last question in this scope: drop its stale answers."""
        old = self._kb_seen.get(scope)
        self._kb_seen[scope] = kb
        if old is not None and old != kb:
            try:
                self.backend.purge({"scope": scope})
            except Exception as e:
                self.last_error = f"purge: {e}"[:200]

    def answer(self, question: str, scope: str, kb: str, compute: Callable[[], tuple[str, int]]) -> dict:
        """Cached answer for `question`, or compute() -> (answer, tokens_used) and store it.
        Returns {answer, cache: {...}}. Cache failures never block the answer."""
        attrs = {"scope": scope, "kb": kb}
        self._note_kb(scope, kb)
        t0 = time.time()
        hit = None
        try:
            hit = self.backend.search(question, attrs, self.threshold)
        except Exception as e:
            with self.lock:
                self.stats["errors"] += 1
            self.last_error = f"search: {e}"[:200]
        if hit and hit.get("response"):
            saved = int(hit.get("tokens") or max(1, (len(question) + len(hit["response"]) + 1500) // 4))
            with self.lock:
                self.stats["hits"] += 1
                self.stats["tokens_saved"] += saved
            return {"answer": hit["response"], "cache": {
                "hit": True, "backend": self.backend_name, "similarity": round(hit["similarity"], 3),
                "matched_question": hit.get("prompt"), "tokens_saved": saved,
                "seconds": round(time.time() - t0, 3)}}
        t1 = time.time()
        answer, tokens = compute()
        llm_s = time.time() - t1
        with self.lock:
            self.stats["misses"] += 1
        stored = False
        if answer:
            try:
                self.backend.set(question, answer, attrs, tokens)
                stored = True
            except Exception as e:
                with self.lock:
                    self.stats["errors"] += 1
                self.last_error = f"store: {e}"[:200]
        return {"answer": answer, "cache": {"hit": False, "backend": self.backend_name, "stored": stored,
                                            "tokens_used": tokens, "seconds": round(time.time() - t0, 2),
                                            "llm_seconds": round(llm_s, 2)}}

    def summary(self) -> dict:
        with self.lock:
            s = dict(self.stats)
        total = s["hits"] + s["misses"]
        return {"backend": self.backend_name, "threshold": self.threshold, **s,
                "hit_rate": round(s["hits"] / total, 2) if total else None, "last_error": self.last_error}


_shared: Optional[SemanticCache] = None
_shared_lock = threading.Lock()


def shared_cache() -> SemanticCache:
    """One cache per process, shared by the exam and proctoring Q&A."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = SemanticCache.from_env()
            print(f"[cache] semantic cache: {_shared.backend_name} (threshold {_shared.threshold})", flush=True)
        return _shared
