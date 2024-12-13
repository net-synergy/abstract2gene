"""Download and strip pubtator's BioCXML files.

Files are large tarball that include abstracts for all pubmed publications and
full text for many.

Downloading and processing files takes a long time since each tarball is ~16GB
compressed and must be decompressed.
"""

__all__ = ["BiocDownloader"]

import re
from typing import Iterable

from ._downloader import FtpDownloader

_FTP_INFO = {
    "server": "ftp.ncbi.nlm.nih.gov",
    "dir": "pub/lu/PubTator3",
}

_BIOC_TEMPLATE = "BioCXML.{}.tar.gz"
_NAME = "bioc"


class BiocDownloader(FtpDownloader):
    """Download gene files from Pubtator FTP server."""

    def __init__(self, cache_dir: str | None = None, prompt: bool = True):
        self.prompt = prompt
        self.file_numbers = list(range(10))
        super().__init__(self.files, cache_dir)

    @property
    def name(self) -> str:
        return _NAME

    @property
    def server(self) -> str:
        return "ftp.ncbi.nlm.nih.gov"

    @property
    def subdir(self) -> str:
        return "/pub/lu/PubTator3"

    @property
    def file_numbers(self) -> list[int]:
        return self._file_numbers

    @file_numbers.setter
    def file_numbers(self, numbers: int | Iterable[int]):
        if isinstance(numbers, int):
            self._file_numbers = [numbers]
        else:
            self._file_numbers = list(numbers)

        self.files = [_BIOC_TEMPLATE.format(i) for i in self._file_numbers]

    def download(self) -> list[str]:
        msg = f"""
        The BioCXML files are ~16GB each for a total of {16 *
        len(self.file_numbers)}GB total (compressed). It may be possible and
        preferred to only download a subset of all files. To download a subset
        of files set this class's `file_numbers` to the files desired then
        rerun download.

        Download all files? (y/N)
        """

        all_files = len(set(range(10)).difference(set(self.file_numbers))) == 0
        if self.prompt and all_files and input(msg).lower() != "y":
            raise RuntimeError("Download canceled by user.")

        return super().download()
