import os

import appdirs

from abstract2gene import __name__ as pkg_name

_APPAUTHOR = "net_synergy"


def default_cache_dir() -> str:
    """Find the default location to save cache files.

    If the directory does not exist it is created.

    Cache files are specifically files that can be easily reproduced,
    i.e. those that can be downloaded from the internet.
    """
    cache_dir = appdirs.user_cache_dir(pkg_name, _APPAUTHOR)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir, mode=0o755)

    return cache_dir


def default_data_dir() -> str:
    """Find the default location to save data files.

    If the directory does not exist it is created.

    Data files are files created by a user. It's possible they can be
    reproduced by rerunning the script that produced them but there is
    no guarantee they can be perfectly reproduced.
    """
    data_dir = appdirs.user_data_dir(pkg_name, _APPAUTHOR)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir, mode=0o755)

    return data_dir
