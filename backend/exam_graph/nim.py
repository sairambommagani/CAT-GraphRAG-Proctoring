"""NVIDIA NIM text models for the exam knowledge graph (question generation, verification,
document structuring, concept relations, examiner answers).

Model chain (EXAM_LLM_MODELS, first available wins, same fallback logic as the judge):
  1. nvidia/llama-3.3-nemotron-super-49b-v1.5  NVIDIA Nemotron Super: strong reasoning and
                                               instruction following, good at exam-quality MCQs
  2. meta/llama-3.3-70b-instruct               widely available fallback

Nemotron Super reasons by default; the system prompt "/no_think" turns that off for the
JSON-producing calls (faster, fewer tokens). Shares the NVIDIA key, RPM limiter and daily
budget style of the proctoring judge.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import httpx

from proctor.judge import RateLimiter, UsageTracker, _json_objects

DEFAULT_MODELS = ("nvidia/llama-3.3-nemotron-super-49b-v1.5", "meta/llama-3.3-70b-instruct")

# When the configured models are gone (404/410 - NVIDIA retires models from the free API),
# the client looks in the key's model catalogue for these, in order.
TEXT_PREFERENCES = ("nvidia/nemotron-3-super-120b-a12b", "nvidia/llama-3.3-nemotron-super-49b-v1.5",
                    "nvidia/llama-3.1-nemotron-ultra-253b-v1", "nvidia/nemotron-3-nano-30b-a3b",
                    "meta/llama-4-maverick-17b-128e-instruct", "moonshotai/kimi-k2-instruct",
                    "qwen/qwen3-next-80b-a3b-instruct", "deepseek-ai/deepseek-v3.1",
                    "mistralai/mistral-medium-3-instruct", "meta/llama-3.3-70b-instruct",
                    "meta/llama-3.1-70b-instruct")
SKIP_WORDS = ("embed", "rerank", "reward", "guard", "safety", "vl", "vision", "retriever", "parse", "ocr",
              "omni", "cosmos", "clip", "asr", "tts", "translate", "code", "coder", "math", "pii")


def rank_text_models(catalogue: list[str]) -> list[str]:
    """Chat-capable text models from the key's catalogue, best first."""
    cat = set(catalogue)
    ranked = [m for m in TEXT_PREFERENCES if m in cat]
    extra = [m for m in sorted(cat) if m not in ranked and not any(w in m.lower() for w in SKIP_WORDS)
             and any(k in m.lower() for k in ("nemotron", "llama", "kimi", "qwen", "deepseek", "mistral", "gemma"))
             and ("instruct" in m.lower() or "nemotron" in m.lower() or "kimi" in m.lower())]
    return ranked + extra


class LLMUnavailable(RuntimeError):
    pass


class NIMText:
    def __init__(self, models=None, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 client: Optional[httpx.Client] = None, rpm: int = 30, usage_path: Optional[str] = None,
                 max_retries: int = 3):
        env = os.environ.get
        chain = [m.strip() for m in (env("EXAM_LLM_MODELS") or "").split(",") if m.strip()]
        self.models = list(models or chain or DEFAULT_MODELS)
        self.api_key = api_key if api_key is not None else (env("NVIDIA_API_KEY") or env("NIM_API_KEY"))
        self.base_url = (base_url or env("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")).rstrip("/")
        self.client = client or httpx.Client(timeout=120)
        self.limiter = RateLimiter(rpm)
        self.usage = UsageTracker(usage_path if usage_path is not None else env("EXAM_LLM_USAGE_PATH"),
                                  int(env("EXAM_LLM_DAILY_CALLS", "400")), int(env("EXAM_LLM_DAILY_TOKENS", "2000000")))
        self.max_retries = max_retries
        self.down: dict[str, float] = {}
        self.last_model: Optional[str] = None
        self.last_tokens = 0
        self._discovered = False

    @property
    def available(self) -> bool:
        return bool(self.api_key) and not str(self.api_key).startswith("nvapi-xxx")

    def _post(self, payload: dict) -> dict:
        delay = 2.0
        for attempt in range(self.max_retries):
            self.limiter.acquire()
            r = self.client.post(f"{self.base_url}/chat/completions", json=payload,
                                 headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"})
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                time.sleep(min(float(r.headers.get("retry-after", delay)), 30))
                delay *= 2
                continue
            raise httpx.HTTPStatusError(f"NIM {r.status_code}: {r.text[:200]}", request=r.request, response=r)
        raise RuntimeError("unreachable")

    def catalogue(self) -> list[str]:
        try:
            r = self.client.get(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.api_key}"}, timeout=30)
            r.raise_for_status()
            return sorted(m["id"] for m in r.json().get("data", []))
        except Exception:
            return []

    def _discover(self) -> None:
        """Configured models all unavailable: add the best ones this key actually has."""
        self._discovered = True
        found = [m for m in rank_text_models(self.catalogue()) if m not in self.models][:4]
        if found:
            print(f"[exam-graph] configured text models unavailable; using {found[0]} from your NVIDIA catalogue",
                  flush=True)
            self.models += found

    def chat(self, system: str, user: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
        try:
            return self._chat(system, user, max_tokens, temperature)
        except LLMUnavailable:
            if self._discovered or not self.available:
                raise
            self._discover()
            return self._chat(system, user, max_tokens, temperature)

    def _chat(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        if not self.available:
            raise LLMUnavailable("NVIDIA_API_KEY is not set")
        if not self.usage.can_spend():
            raise LLMUnavailable("daily NIM budget for the exam generator reached")
        errors = []
        for model in self.models:
            if time.time() < self.down.get(model, 0):
                continue
            sys_prompt = ("/no_think\n" + system) if "nemotron" in model and "super" in model else system
            try:
                resp = self._post({"model": model, "temperature": temperature, "top_p": 0.95,
                                   "max_tokens": max_tokens, "stream": False,
                                   "messages": [{"role": "system", "content": sys_prompt},
                                                {"role": "user", "content": user}]})
            except httpx.HTTPStatusError as e:
                code = e.response.status_code if e.response is not None else 0
                self.down[model] = time.time() + (3600 if code in (401, 403, 404, 410) else 120)
                errors.append(f"{model}: HTTP {code}")
                continue
            except (httpx.TimeoutException, httpx.TransportError) as e:
                self.down[model] = time.time() + 120
                errors.append(f"{model}: {type(e).__name__}")
                continue
            self.last_tokens = int((resp.get("usage") or {}).get("total_tokens", 0))
            self.usage.record(self.last_tokens)
            self.last_model = model
            text = resp["choices"][0]["message"].get("content") or ""
            return re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.S).strip()
        raise LLMUnavailable("no exam LLM available - " + "; ".join(errors))

    def chat_json(self, system: str, user: str, **kw):
        """Last JSON object/array in the reply."""
        text = self.chat(system, user, **kw)
        text = re.sub(r"```(?:json)?", "", text)
        arr = _last_json_array(text)
        if arr is not None:
            return arr
        for cand in reversed(_json_objects(text)):
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                continue
        raise ValueError(f"no JSON in model reply: {text[:200]!r}")


def _last_json_array(text: str):
    depth, start, found = 0, -1, None
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
            if depth == 0:
                try:
                    val = json.loads(text[start:i + 1])
                    if isinstance(val, list) and (not val or isinstance(val[0], dict)):
                        found = val
                except json.JSONDecodeError:
                    pass
    return found
