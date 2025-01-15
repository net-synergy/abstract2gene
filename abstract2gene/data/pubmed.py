"""Download and manage data from Pubmed."""

__all__ = [
    "PubmedDownloader",
    "list_cache",
    "delete_from_cache",
]

import os

import numpy as np
import pandas as pd

from abstract2gene.storage import _storage_factory, default_cache_dir
from abstract2gene.storage import delete_from_cache as _delete_cache
from abstract2gene.storage import list_cache as _list_cache

from ._downloader import FtpDownloader

_NAME = "pubmed"
_FILES = ["gene2pubmed.gz", "gene_info.gz"]

list_cache = _storage_factory(_list_cache, _NAME)
delete_from_cache = _storage_factory(_delete_cache, _NAME)


class PubmedDownloader(FtpDownloader):
    """Download gene files from Pubmed FTP server."""

    def __init__(self, **kwds):
        super().__init__(_FILES, **kwds)

    @property
    def name(self) -> str:
        return _NAME

    @property
    def server(self) -> str:
        return "ftp.ncbi.nlm.nih.gov"

    @property
    def subdir(self) -> str:
        return "/gene/DATA/"
