"""Download and manage data from Pubtator3."""

__all__ = [
    "PubtatorDownloader",
    "read_pubtator_data",
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

_NAME = "pubtator"
_FILES = ["gene2pubtator3.gz"]

list_cache = _storage_factory(_list_cache, _NAME)
delete_from_cache = _storage_factory(_delete_cache, _NAME)


class PubtatorDownloader(FtpDownloader):
    """Download gene files from Pubtator FTP server."""

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
        return "/pub/lu/PubTator3"


def read_pubtator_data(edge_path) -> pd.DataFrame:
    table = pd.read_table(
        edge_path,
        header=None,
        names=["PMID", "Type", "NCBIGeneID", "GeneSymbol", "Resource"],
        usecols=["PMID", "NCBIGeneID", "GeneSymbol"],
        memory_map=True,
        low_memory=False,
    )
    table = table[pd.notna(table["NCBIGeneID"])]
    table["NCBIGeneID"] = (
        table["NCBIGeneID"]
        .astype("str")
        .map(lambda x: x.split(";")[0])
        .astype("int64")
    )

    return table
