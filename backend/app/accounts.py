"""Candidate accounts: register / sign in with a username + password.

Passwords are stored as PBKDF2-SHA256 hashes (200k iterations, per-user salt) in
data/accounts.json; sign-in returns a random bearer token kept in memory (a server
restart signs everyone out). Demo-scale: swap for the platform's identity provider
(SSO / OAuth) in production.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

_PATH = Path(os.environ.get("CAT_ACCOUNTS_PATH", "data/accounts.json"))
_lock = threading.Lock()
_tokens: dict[str, dict] = {}
TOKEN_TTL_S = 12 * 3600


def _load() -> dict:
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(users: dict) -> None:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(users, indent=1), encoding="utf-8")
    tmp.replace(_PATH)


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()


def register(username: str, password: str, display_name: str = "") -> dict:
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.@-]{3,40}", username):
        raise ValueError("username: 3-40 letters, digits, . _ - @")
    if len(password) < 6:
        raise ValueError("password: at least 6 characters")
    with _lock:
        users = _load()
        if username in users:
            raise ValueError("this username is already registered - sign in instead")
        salt = secrets.token_bytes(16)
        users[username] = {"name": (display_name or username).strip()[:60], "salt": salt.hex(),
                           "hash": _hash(password, salt), "created_at": time.time()}
        _save(users)
    return login(username, password)


def login(username: str, password: str) -> dict:
    username = username.strip().lower()
    u = _load().get(username)
    if not u or not hmac.compare_digest(u["hash"], _hash(password, bytes.fromhex(u["salt"]))):
        raise PermissionError("wrong username or password")
    token = secrets.token_urlsafe(24)
    with _lock:
        _tokens[token] = {"username": username, "name": u["name"], "expires": time.time() + TOKEN_TTL_S}
    return {"token": token, "username": username, "name": u["name"]}


def user_for(token: Optional[str]) -> Optional[dict]:
    if not token:
        return None
    t = _tokens.get(token)
    if not t or t["expires"] < time.time():
        return None
    return {"username": t["username"], "name": t["name"]}
