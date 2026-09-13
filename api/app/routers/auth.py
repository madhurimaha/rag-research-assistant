"""Signup, login, current user, logout."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException

from app.core.auth import (
    bearer_token,
    create_session,
    delete_session,
    get_current_user,
    hash_password,
    verify_password,
)
from app.db.pool import connection
from app.schemas.models import AuthOut, LoginRequest, SignupRequest, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


def _normalize_email(email: str) -> str:
    return email.strip().lower()


@router.post("/signup", response_model=AuthOut)
def signup(req: SignupRequest) -> AuthOut:
    email = _normalize_email(req.email)
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(400, "enter a valid email address")
    with connection() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = %s", (email,)).fetchone()
        if existing:
            raise HTTPException(409, "an account with that email already exists")
        user = conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id, email",
            (email, hash_password(req.password)),
        ).fetchone()
    token = create_session(user["id"])
    return AuthOut(token=token, user=UserOut(**user))


@router.post("/login", response_model=AuthOut)
def login(req: LoginRequest) -> AuthOut:
    email = _normalize_email(req.email)
    with connection() as conn:
        user = conn.execute(
            "SELECT id, email, password_hash FROM users WHERE email = %s", (email,)
        ).fetchone()
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "email or password is wrong")
    token = create_session(user["id"])
    return AuthOut(token=token, user=UserOut(id=user["id"], email=user["email"]))


@router.get("/me", response_model=UserOut)
def me(user: dict = Depends(get_current_user)) -> UserOut:
    return UserOut(**user)


@router.post("/logout", status_code=204)
def logout(authorization: str | None = Header(default=None)) -> None:
    token = bearer_token(authorization)
    if token:
        delete_session(token)
