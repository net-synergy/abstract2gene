"""Authenticate a user."""

__ALL__ = ["is_authorized", "router", "get_token_from_cookie"]

import json
import os
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext


def _get_users() -> dict[str, Any]:
    user_file = (
        "auth/users.json"
        if os.path.exists("auth/users.json")
        else os.path.join(
            os.getenv("HOME") or "/root", ".local/share/auth/users.json"
        )
    )

    with open(user_file, "r") as f:
        return json.load(f)


users = _get_users()

router = APIRouter()
security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def is_authorized(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> bool:
    if (credentials.username not in users) or (
        not pwd_context.verify(
            credentials.password, users[credentials.username]
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return True
