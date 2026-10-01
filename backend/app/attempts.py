"""Per-candidate attempt history (data/attempts.json), so performance carries across exams:
the next exam on a subject starts from the last ability estimate and targets the concepts
missed last time; the result page shows progress across attempts."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

_PATH = Path(os.environ.get("CAT_ATTEMPTS_PATH", "data/attempts.json"))
_lock = threading.Lock()


def _load() -> dict:
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def record(username: str, attempt: dict) -> None:
    with _lock:
        data = _load()
        data.setdefault(username, []).append({**attempt, "finished_at": time.time()})
        data[username] = data[username][-50:]
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(_PATH)


def history(username: Optional[str], subject: Optional[str] = None) -> list[dict]:
    if not username:
        return []
    rows = _load().get(username, [])
    return [r for r in rows if subject is None or r.get("subject") == subject]


def last(username: Optional[str], subject: str) -> Optional[dict]:
    h = history(username, subject)
    return h[-1] if h else None
