"""
JWT Bearer Authentication Dependency
======================================
FastAPI dependency that validates a Bearer JWT token and returns the
decoded user payload.  Import `CurrentUser` in any route that requires
authentication:

    from app.middleware.auth import CurrentUser

    @router.get("/protected")
    async def protected_route(user: CurrentUser):
        return {"username": user["username"]}
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from config.settings import settings

_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> dict:
    """Validate Bearer JWT and return decoded user payload."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated — provide a Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt.jwt_secret_key,
            algorithms=[settings.jwt.jwt_algorithm],
        )
        username: str = payload.get("sub", "")
        if not username:
            raise ValueError("Token missing subject")
        return {
            "username": username,
            "role": payload.get("role", "viewer"),
        }
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Convenient type alias — use as a route parameter type
CurrentUser = Annotated[dict, Depends(get_current_user)]
