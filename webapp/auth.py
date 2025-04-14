"""Authenticate a user."""

__ALL__ = ["is_authorized", "router", "get_token_from_cookie"]

import json
import os
import time
from typing import Annotated, Any

import bcrypt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from webapp import config as cfg


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
security = (
    HTTPBasic()
    if cfg.use_auth
    else lambda: HTTPBasicCredentials(username="", password="")
)


async def is_authorized(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> bool:
    if not cfg.use_auth:
        return True

    entered_passwd = credentials.password.encode("utf-8")
    if (credentials.username not in users) or (
        not bcrypt.checkpw(
            entered_passwd, users[credentials.username].encode("utf-8")
        )
    ):

        time.sleep(3)
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return True
