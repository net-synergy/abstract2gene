"""Base class for downloading FTP files."""

__all__ = ["FtpDownloader"]

import datetime
import os
from ftplib import FTP
from typing import Iterable

from tqdm import tqdm

from abstract2gene.storage import default_cache_dir


class FtpDownloader:
    """A base class for downloading files from FTP servers."""

    def __init__(
        self,
        files: Iterable[str],
        cache_dir: str | None = None,
    ):
        self.files = files
        self.cache_dir = cache_dir or default_cache_dir(self.name)
        self.ftp: FTP | None = self.connect()
        self.ls = list(self.ftp.mlsd(facts=["modify"]))

    @property
    def name(self) -> str:
        """Name of the downloader (where to store downloads)."""
        raise NotImplementedError

    @property
    def server(self) -> str:
        """Name of the server to download from."""
        raise NotImplementedError

    @property
    def subdir(self) -> str:
        """Name of subdirectory files are located."""
        raise NotImplementedError

    def connect(self) -> FTP:
        ftp = FTP(self.server)
        ftp.login()
        ftp.cwd(self.subdir)

        return ftp

    def disconnect(self) -> None:
        if self.ftp is None:
            return

        self.ftp.close()
        self.ftp = None

    def _local(self, file: str) -> str:
        """Location where the local file should be store."""
        return os.path.join(self.cache_dir, file)

    def _download_file(self, file: str) -> str:
        """Download a single file from the server."""
        if self.ftp is None:
            # _download_file is only called by download which ensures
            # connection exists so shouldn't matter but to keep mypy happy and
            # just in case double check self.ftp is set.
            raise RuntimeError("FTP not connected")

        remote_files = [f[0] for f in self.ls]
        if file not in remote_files:
            raise ValueError(f'File "{file}" not found on server.')

        if os.path.exists(self._local(file)):
            if self._is_old(file):
                os.unlink(self._local(file))
            else:
                print(f"Most recent file for {file} already downloaded.")
                return self._local(file)

        # Needed to switch from ASCII to binary mode to work with ftp.size.
        self.ftp.voidcmd("TYPE I")
        keys = {
            "total": self.ftp.size(file),
            "desc": file,
            "unit": "B",
            "unit_scale": True,
        }
        with open(self._local(file), "wb") as fp, tqdm(**keys) as pbar:

            def progress_callback(data):
                fp.write(data)
                pbar.update(len(data))

            self.ftp.retrbinary(f"RETR {file}", progress_callback)

        return self._local(file)

    def _is_old(self, file: str) -> bool:
        def _remote_modification(file: str) -> datetime.datetime:
            file_info = [f[1] for f in self.ls if f[0] == file][0]
            mod_string = file_info["modify"]
            mod_string = mod_string[:8] + "T" + mod_string[8:]
            return datetime.datetime.fromisoformat(mod_string)

        local_modtime = datetime.datetime.fromtimestamp(
            os.stat(self._local(file)).st_mtime
        )

        return local_modtime < _remote_modification(file)

    def download(self) -> list[str]:
        """Download the requested files."""
        if self.ftp is None:
            self.ftp = self.connect()

        print("Starting downloads:")
        files = [self._download_file(file) for file in self.files]
        self.disconnect()

        return files
