"""Download and manage data from Pubtator3.

For attaching publication annotations to PubNet objects.
"""

__all__ = [
    "PubtatorDownloader",
    "read_pubtator_data",
    "add_gene_edges",
    "list_cache",
    "delete_from_cache",
]

import os

import numpy as np
import pandas as pd
from pubnet import PubNet

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


def add_gene_edges(
    net: PubNet,
    replace: bool = False,
    cache_dir: str | None = None,
) -> None:
    """Add edges from a pubtator gene file to the provided PubNet.

    Parameters
    ----------
    net : PubNet
        The net to attach the gene nodes and edges to.
    replace : bool, optional
        What to do if Gene--Publication nodes already exist in `net`. By
        default (false) raise an error. If replace is true, remove the old
        edges and nodes.
    cache_dir : str, optional
        Where to search for downloaded edge files (default's to
        `default_cache_dir()`).

    """
    cache_dir = cache_dir or default_cache_dir()
    if "Gene-Publication" in net.edges:
        if replace:
            net.drop(nodes=("Gene",), edges=(("Gene", "Publication"),))
        else:
            raise RuntimeError("Gene--Publication edges already in net.")

    file = os.path.join(_NAME, _FILES[0])
    edge_path = os.path.join(cache_dir, file)
    if not os.path.exists(edge_path):
        msg = f"{_FILES[0]} has not been downloaded, download now? (Y/n)"
        if input(msg).lower() == "n":
            raise RuntimeError("Need to download data files.")

        downloader = PubtatorDownloader(cache_dir=cache_dir)
        downloader.download()

    table = read_pubtator_data(edge_path)
    node_data = table[
        [col for col in table.columns if col not in ("PMID", "TaxID")]
    ]
    node_data = node_data.drop_duplicates(
        ignore_index=True, subset=["NCBIGeneID"]
    )

    net.add_node(node_data, "Gene")

    pub_node = net.get_node("Publication")
    pmid2pubid = dict(zip(pub_node.feature_vector("PMID"), pub_node.index))
    ncbi2geneid = {row.NCBIGeneID: row.Index for row in node_data.itertuples()}

    edge_data = np.fromiter(
        (
            (pmid2pubid[row.PMID], ncbi2geneid[row.NCBIGeneID])
            for row in table[["PMID", "NCBIGeneID"]].itertuples(index=False)
            if row.PMID in pmid2pubid
        ),
        dtype=np.dtype((int, 2)),
    )
    net.add_edge(edge_data, start_id="Publication", end_id="Gene")
