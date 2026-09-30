import json
from pathlib import Path
from functools import lru_cache

_BANK_PATH = Path(__file__).parent / "data" / "question_bank.json"


@lru_cache(maxsize=1)
def load_bank() -> list[dict]:
    with open(_BANK_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def bank_by_id() -> dict[str, dict]:
    return {q["id"]: q for q in load_bank()}
