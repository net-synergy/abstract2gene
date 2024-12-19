"""Download and strip pubtator's BioCXML files.

Files are large tarball that include abstracts for all pubmed publications and
full text for many.

Downloading and processing files takes a long time since each tarball is ~16GB
compressed and must be decompressed.
"""

__all__ = [
    "BiocDownloader",
    "list_cache",
    "delete_from_cache",
]

from typing import Iterable

from abstract2gene.storage import _storage_factory
from abstract2gene.storage import delete_from_cache as _delete_cache
from abstract2gene.storage import list_cache as _list_cache

from ._downloader import FtpDownloader

_FTP_INFO = {
    "server": "ftp.ncbi.nlm.nih.gov",
    "dir": "pub/lu/PubTator3",
}

_BIOC_TEMPLATE = "BioCXML.{}.tar.gz"
_NAME = "bioc"

list_cache = _storage_factory(_list_cache, _NAME)
delete_from_cache = _storage_factory(_delete_cache, _NAME)


class BiocDownloader(FtpDownloader):
    """Download gene files from Pubtator FTP server."""

    def __init__(self, **kwds):
        self.file_numbers = list(range(10))
        super().__init__(self.files, **kwds)

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
    def file_numbers(self) -> Iterable[int]:
        return self._file_numbers

    @file_numbers.setter
    def file_numbers(self, numbers: int | Iterable[int]):
        if isinstance(numbers, int):
            self._file_numbers = {numbers}
        else:
            self._file_numbers = set(numbers)

        self.files = [_BIOC_TEMPLATE.format(i) for i in self._file_numbers]
