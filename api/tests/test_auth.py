"""Signup, login, protected routes, and conversation isolation.

Uses a slim FastAPI app (routers only, no model warmup) against the same Postgres as local
dev. Skips when the database is not reachable so unit tests still run offline.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.db.pool import close_pool, connection, init_schema
from app.routers import auth, chat


@pytest.fixture
def client() -> TestClient:
    try:
        init_schema()
    except Exception:
        pytest.skip("postgres is not running")
    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(chat.router)
    yield TestClient(app)
    close_pool()


def _account(client: TestClient) -> tuple[str, dict]:
    email = f"iso-{uuid4().hex[:12]}@example.com"
    password = "testdemo1"
    res = client.post("/auth/signup", json={"email": email, "password": password})
    assert res.status_code == 200, res.text
    body = res.json()
    return body["token"], body["user"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class TestAuth:
    def test_signup_then_login_then_me(self, client: TestClient) -> None:
        token, user = _account(client)
        me = client.get("/auth/me", headers=_auth(token))
        assert me.status_code == 200
        assert me.json()["email"] == user["email"]

        login = client.post(
            "/auth/login", json={"email": user["email"], "password": "testdemo1"}
        )
        assert login.status_code == 200
        again = client.get("/auth/me", headers=_auth(login.json()["token"]))
        assert again.status_code == 200
        assert again.json()["id"] == user["id"]

    def test_ask_without_token_is_401(self, client: TestClient) -> None:
        res = client.post("/chat/ask", json={"question": "What language are the tweets?"})
        assert res.status_code == 401

    def test_conversation_is_invisible_to_another_account(self, client: TestClient) -> None:
        token_a, user_a = _account(client)
        token_b, _user_b = _account(client)

        with connection() as conn:
            convo_id = conn.execute(
                "INSERT INTO conversations (user_id, title) VALUES (%s, %s) RETURNING id",
                (user_a["id"], "only for A"),
            ).fetchone()["id"]

        as_a = client.get(f"/conversations/{convo_id}", headers=_auth(token_a))
        assert as_a.status_code == 200
        assert as_a.json()["id"] == convo_id

        as_b = client.get(f"/conversations/{convo_id}", headers=_auth(token_b))
        assert as_b.status_code == 404

        listed = client.get("/conversations", headers=_auth(token_b))
        assert listed.status_code == 200
        assert convo_id not in {row["id"] for row in listed.json()}
