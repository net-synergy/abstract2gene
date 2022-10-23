import os
import sys

from abstract2gene import __name__ as pkg_name


def default_cache_dir():
    """Find the default location to save cache files.

    Cache files are specifically files that can be easily reproduced,
    i.e. those that can be downloaded from the internet.
    """

    if sys.platform.startswith("win"):
        try:
            cache_home = os.environ["LOCALAPPDATA"]
        except KeyError as err:
            raise EnvironmentError(
                "Location for local app data is not set.",
                "Explicitely set cache directory to get around error.",
            ) from err
    else:
        try:
            cache_home = os.environ["XDG_CACHE_HOME"]
        except KeyError:
            home = os.environ["HOME"]
            return os.path.join(home, pkg_name, "cache")

    return os.path.join(cache_home, pkg_name)


def default_data_dir():
    """Find the default location to save data files.

    Data files are files created by a user. It's possible they can be
    reproduced by rerunning the script that produced them but there is
    no gurentee they can be perfectly reproduced.
    """

    if sys.platform.startswith("win"):
        try:
            data_home = os.environ["APPDATA"]
        except KeyError as err:
            raise EnvironmentError(
                "Location for app data is not set.",
                "Explicitely set cache directory to get around error.",
            ) from err
    else:
        try:
            data_home = os.environ["XDG_DATA_HOME"]
        except KeyError:
            home = os.environ["HOME"]
            return os.path.join(home, pkg_name, "share")

    return os.path.join(data_home, pkg_name)
