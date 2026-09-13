"""Session-based auth.

Email + password, sessions stored in Postgres. No JWT library: a random token in `sessions` is
enough for a local multi-user demo, and a row we can delete is a real logout.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Header, HTTPException

from app.db.pool import connection

_ITERATIONS = 480_000
_SESSION_DAYS = 14


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _ITERATIONS)
    return f"pbkdf2$sha256${_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, algo, iters, salt, digest = stored.split("$")
        if scheme != "pbkdf2" or algo != "sha256":
            return False
        check = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(check.hex(), digest)
    except (ValueError, TypeError):
        return False


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=_SESSION_DAYS)
    with connection() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (%s, %s, %s)",
            (token, user_id, expires),
        )
    return token


def delete_session(token: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM sessions WHERE token = %s", (token,))


def user_for_token(token: str) -> dict | None:
    with connection() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.email
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = %s AND s.expires_at > now()
            """,
            (token,),
        ).fetchone()
    return dict(row) if row else None


def bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    kind, _, value = authorization.partition(" ")
    if kind.lower() != "bearer" or not value:
        return None
    return value


def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    token = bearer_token(authorization)
    if not token:
        raise HTTPException(401, "sign in required")
    user = user_for_token(token)
    if not user:
        raise HTTPException(401, "session expired")
    return user
