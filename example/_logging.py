"""Set up logging to print information to files.

Intended to store all information relevant to the methods and results section.
"""

__ALL__ = ["set_log", "log"]

import os
from datetime import datetime

LOGDIR = "results"
_logfile: str | None = None


def set_log(name: str):
    global _logfile

    if not os.path.exists(LOGDIR):
        os.mkdir(LOGDIR)

    if not os.path.exists(os.path.join(LOGDIR, name)):
        os.mkdir(os.path.join(LOGDIR, name))

    _logfile = os.path.join(LOGDIR, name, f"{datetime.today()}.log")


def log(message: str):
    if not _logfile:
        raise RuntimeError("Log file not set. Set with `set_log`.")

    with open(_logfile, "a") as f:
        print(message, file=f)
