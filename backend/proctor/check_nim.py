"""Test the AI-judge model chain against your NVIDIA API key.

    python -m proctor.check_nim

1. Lists the models your key can use (GET /v1/models, free) and marks the
   vision/audio-capable ones.
2. Sends one small synthetic clip to EACH model in NIM_MODELS (default:
   Cosmos 3 Reasoner -> Nemotron 3 Omni -> Llama 3.2 Vision) with its richest
   input format, stepping down on 400s, and prints which input works, the
   verdict, the reasoning, latency and tokens. Nemotron Omni also gets a
   synthetic audio track.
3. Saves what works to data/nim_models.json, so the server starts with the
   right model and input format, and prints the NIM_MODELS line to use.

Costs one call per model (~2-4k tokens each).
"""
from __future__ import annotations

import sys
import time

import cv2
import httpx
import numpy as np

from .clip import BufferedFrame, Evidence
from .detector import Event
from .judge import JudgeConfig, ModelUnavailable, NIMJudge, RateLimiter, profile

VISION_HINTS = ("vl", "vision", "multimodal", "omni", "cosmos", "reason", "gemma-3", "llama-4",
                "phi-4-multimodal", "kimi-k", "glimmer", "pixtral")
SKIP_HINTS = ("embed", "retriever", "90b")


def _test_evidence() -> Evidence:
    """A person-like figure whose head drops toward a dark 'phone' rectangle, 16 frames over
    4 s, with a short synthetic voice in the audio track (so Omni's audio input is exercised)."""
    frames = []
    for i in range(16):
        img = np.full((336, 448, 3), (170, 150, 120), np.uint8)
        dy = 0 if i < 6 else 40
        cv2.ellipse(img, (224, 330), (130, 90), 0, 180, 360, (60, 60, 160), -1)
        cv2.circle(img, (224, 150 + dy), 55, (140, 170, 220), -1)
        if i >= 6:
            cv2.rectangle(img, (190, 275), (260, 330), (30, 30, 30), -1)
        cv2.putText(img, f"t={i * 0.25:.2f}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        frames.append(BufferedFrame(i * 0.25, cv2.imencode(".jpg", img)[1].tobytes(), {"face_count": 1}))
    ev = Event("OFFSCREEN_SUSTAINED", 1.5, 4.0, "down", score=1.6, details={"duration_s": 2.5})
    t = np.arange(16000 * 4) / 16000
    audio = (np.sin(2 * np.pi * 140 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) * 6000).astype(np.int16)
    audio[: 16000 * 2] = 0
    out = Evidence(ev, frames[::2], 0.0, 4.0, video_frames=frames, audio=audio, speech_s=2.0,
                   timeline=["on-screen 0.0-1.4s", "off-screen down 1.5-4.0s (peak 1.6)", "voice 2.0-4.0s"])
    out.knowledge = {"rules": [{"id": "R-02", "text": "Looking down at the desk or lap: reading a phone or notes "
                                                      "below the camera is a violation."}], "cases": []}
    return out


def list_models(cfg: JudgeConfig, client: httpx.Client) -> list[str]:
    try:
        r = client.get(cfg.base_url.rstrip("/") + "/models",
                       headers={"Authorization": f"Bearer {cfg.api_key}"}, timeout=30)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))
    except Exception as e:
        print(f"(could not list models: {e})")
        return []


def main() -> int:
    cfg = JudgeConfig.from_env()
    print("judge chain      : " + " -> ".join(cfg.models))
    print(f"endpoint         : {cfg.chat_url or cfg.base_url}")
    if not cfg.api_key or cfg.api_key.startswith("nvapi-xxx"):
        print("\nNVIDIA_API_KEY is not set (or still the placeholder). Put your key in backend/.env, "
              "load the .env line in the terminal, and rerun.")
        return 1

    client = httpx.Client(timeout=120)
    models = list_models(cfg, client)
    candidates = list(cfg.models)
    if models:
        print(f"\nyour key lists {len(models)} models:")
        for k in range(0, len(models), 3):
            print("  " + "   ".join(f"{m:<48}" for m in models[k:k + 3]).rstrip())
        for m in cfg.models:
            print(f"  chain model {m}: {'listed' if m in models else 'NOT listed'}")
        # physical-AI (Cosmos) models this key has that the chain doesn't mention yet
        extra = [m for m in models if "cosmos" in m.lower() and "reason" in m.lower() and m not in candidates]
        candidates = [m for m in candidates if "cosmos" in m.lower()] + extra + \
                     [m for m in candidates if "cosmos" not in m.lower()]

    judge = NIMJudge(JudgeConfig(**{**cfg.__dict__, "models": tuple(candidates), "usage_path": None,
                                    "max_retries": 1}), client=client, limiter=RateLimiter(1000))
    for st in judge.state.values():                    # test everything fresh
        st.update(mode=None, mode_audio=None, plain=False, down_until=0.0, last_error="")
    ev = _test_evidence()
    working = []
    print("\ntesting each judge model with a synthetic 'looking down at a phone' clip (+ audio for Omni):")
    for model in candidates:
        t0 = time.monotonic()
        try:
            parsed, usage, mode, reasoning = judge._ask_model(model, ev)
        except ModelUnavailable as e:
            print(f"  x {model}: unavailable - {e}")
            continue
        except ValueError as e:
            print(f"  ? {model}: answered but no usable verdict ({e})")
            continue
        except Exception as e:
            print(f"  x {model}: {type(e).__name__}: {e}")
            continue
        working.append(model)
        print(f"  OK {model}  [used '{mode}'] in {time.monotonic() - t0:.1f}s")
        print(f"     verdict={parsed['verdict']} confidence={parsed['confidence']:.2f} rule={parsed.get('rule') or '-'}"
              f" reason={parsed['reason']!r}")
        if reasoning:
            print(f"     reasoning: {reasoning[:220]}...")
        print(f"     tokens={usage}")
    judge._save_state()

    text_ok = check_exam_models(cfg, client, models)

    print("\n================ RESULT ================")
    if working:
        print(f"judge models that work: {', '.join(working)}")
    else:
        print("no judge model returned a usable verdict")
    print(f"question-generator models that work: {', '.join(text_ok) or 'none'}")
    print("\nPut these lines in backend/.env (replace the old NIM_MODELS / EXAM_LLM_MODELS lines):")
    if working:
        print(f"  NIM_MODELS={','.join(working)}")
    if text_ok:
        print(f"  EXAM_LLM_MODELS={','.join(text_ok)}")
    return 0 if working else 2


def check_exam_models(cfg: JudgeConfig, client: httpx.Client, catalogue: list[str]) -> list[str]:
    """The GraphRAG question generator's NIM text models: configured ones first, then the best
    text models found in the key's catalogue. Stops after two working models."""
    from exam_graph.nim import NIMText, rank_text_models
    llm = NIMText(api_key=cfg.api_key, client=client, usage_path="")
    cands = list(llm.models) + [m for m in rank_text_models(catalogue) if m not in llm.models][:6]
    print("\ntesting the question-generator models (GraphRAG exam graph):")
    ok = []
    for model in cands:
        if len(ok) >= 2:
            break
        one = NIMText(models=[model], api_key=cfg.api_key, client=client, usage_path="")
        one._discovered = True
        t0 = time.monotonic()
        try:
            out = one.chat_json("Reply with a JSON array only.",
                                'Write 1 multiple-choice question about Python lists as '
                                '[{"text": "...", "options": ["a","b","c","d"], "correct_index": 0}]', max_tokens=400)
            q = out[0]
            assert len(q["options"]) == 4
            ok.append(model)
            print(f"  OK {model} in {time.monotonic() - t0:.1f}s: {q['text'][:90]!r}")
        except Exception as e:
            print(f"  x {model}: {str(e)[:160]}")
    return ok


if __name__ == "__main__":
    sys.exit(main())
