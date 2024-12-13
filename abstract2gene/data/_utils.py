__all__ = [
    "default_cache_dir",
    "default_data_dir",
    "list_cache",
    "list_data",
    "delete_from_cache",
    "delete_from_data",
    "storage_factory",
]

import os
from collections.abc import Callable
from typing import TypeVar

import synstore
from typing_extensions import ParamSpec

from abstract2gene import __name__ as pkg_name

synstore.set_package_name(pkg_name)

default_cache_dir = synstore.default_cache_dir
default_data_dir = synstore.default_data_dir
list_cache = synstore.list_cache
list_data = synstore.list_data
delete_from_cache = synstore.delete_from_cache
delete_from_data = synstore.delete_from_data

P = ParamSpec("P")
T = TypeVar("T")


def storage_factory(func: Callable[P, T], subdir: str) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwds: P.kwargs) -> T:
        if len(args) != 0:
            new_args = (os.path.join(subdir, args[0]),) + args[1:]  # type: ignore[call-overload]
        else:
            new_args = args

        if "path" in kwds and kwds["path"] is not None:
            kwds["path"] = os.path.join(subdir, kwds["path"])  # type: ignore[call-overload]
        else:
            kwds["path"] = subdir

        return func(*new_args, **kwds)

    return wrapper
