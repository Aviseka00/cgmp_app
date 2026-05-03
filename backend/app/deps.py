from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .db import db
from .security import decode_token


auth_scheme = HTTPBearer(auto_error=False)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(auth_scheme),
) -> dict[str, Any]:
    if not creds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = decode_token(creds.credentials)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = await db.users.find_one({"username": username})
    if not user or not user.get("is_active", True):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User inactive")
    return user


def require_roles(*allowed_roles: str):
    async def _inner(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        ur = user.get("role")
        if ur not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return user

    return _inner
