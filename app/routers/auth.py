"""
Auth Router
============
POST /auth/token  — exchange username + password for a JWT access token.
POST /auth/verify — check if a token is still valid (used by frontend).

Demo credentials (portfolio):
    admin   / admin123   → role: admin
    analyst / analyst123 → role: analyst
"""

from __future__ import annotations

import time
from typing import Annotated

import bcrypt
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from pydantic import BaseModel

from app.middleware.auth import CurrentUser
from config.settings import settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


def _hash(password: str) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


def _verify(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), hashed)


# In-memory user store — replace with DB in production
_USERS: dict[str, dict] = {
    "admin": {
        "username": "admin",
        "hashed_password": _hash("admin123"),
        "role": "admin",
        "full_name": "Platform Admin",
    },
    "analyst": {
        "username": "analyst",
        "hashed_password": _hash("analyst123"),
        "role": "analyst",
        "full_name": "Data Analyst",
    },
}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    username: str
    role: str


class VerifyResponse(BaseModel):
    valid: bool
    username: str | None = None
    role: str | None = None


@router.post("/token", response_model=TokenResponse)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> TokenResponse:
    """
    Exchange credentials for a JWT access token.

    Form fields: username, password (standard OAuth2 password flow).
    """
    user = _USERS.get(form.username)
    if not user or not _verify(form.password, user["hashed_password"]):
        logger.warning("Login failed", username=form.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expire_seconds = settings.jwt.jwt_expire_minutes * 60
    payload = {
        "sub": user["username"],
        "role": user["role"],
        "exp": int(time.time()) + expire_seconds,
    }
    token = jwt.encode(
        payload,
        settings.jwt.jwt_secret_key,
        algorithm=settings.jwt.jwt_algorithm,
    )

    logger.info("Login successful", username=user["username"], role=user["role"])
    return TokenResponse(
        access_token=token,
        expires_in=expire_seconds,
        username=user["username"],
        role=user["role"],
    )


@router.get("/verify", response_model=VerifyResponse)
async def verify_token(current_user: CurrentUser) -> VerifyResponse:
    """Check if the current Bearer token is valid."""
    return VerifyResponse(
        valid=True,
        username=current_user["username"],
        role=current_user["role"],
    )
