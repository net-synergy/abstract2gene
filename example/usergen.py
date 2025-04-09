"""A CLI to manage users for the abstract2gene web app."""

import argparse
import getpass
import json
import os

from passlib.context import CryptContext

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_FILE = os.path.join(PROJECT_ROOT, "auth", "users.json")


def _read_users() -> dict[str, str]:
    if not os.path.exists(USER_FILE):
        raise FileNotFoundError("The user file does not exist yet.")

    with open(USER_FILE, "r") as f:
        return json.load(f)


def _write_users(users: dict[str, str]) -> None:
    if not os.path.exists(os.path.dirname(USER_FILE)):
        os.mkdir(os.path.dirname(USER_FILE))

    with open(USER_FILE, "w") as f:
        json.dump(users, f)


def ls(args: argparse.Namespace) -> None:
    """List the current users."""
    try:
        users = _read_users()
        if len(users) == 0:
            raise FileNotFoundError()

        print(" ".join(tuple(users.keys())))

    except FileNotFoundError:
        print("No users exist. Use `add` to create a new user.")


def rm(args: argparse.Namespace) -> None:
    """Remove a user."""
    not_found_msg = "User not in database. Nothing to do."
    try:
        users = _read_users()
        if args.user not in users:
            print(not_found_msg)
            return

        users.pop(args.user)
        _write_users(users)

    except FileNotFoundError:
        print(not_found_msg)


def add(args: argparse.Namespace) -> None:
    """Add a new user and create their password."""
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    try:
        users = _read_users()
    except FileNotFoundError:
        users = {}

    if (args.user in users) and (
        input("User exists. Overwrite? (y/N): ").lower() != "y"
    ):
        return

    password = getpass.getpass("Password: ")
    users[args.user] = pwd_context.hash(password)
    _write_users(users)


def main():
    parser = argparse.ArgumentParser(
        prog="usergen", description="Manage users for the web app."
    )
    subparsers = parser.add_subparsers()

    ls_parser = subparsers.add_parser("ls", help="List current users.")
    ls_parser.set_defaults(func=ls)

    rm_parser = subparsers.add_parser("rm", help="Delete a user.")
    rm_parser.add_argument("user", help="Name of the user to remove.")
    rm_parser.set_defaults(func=rm)

    add_parser = subparsers.add_parser("add", help="Add a new user.")
    add_parser.add_argument("user", help="Name of the user to add.")
    add_parser.set_defaults(func=add)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
