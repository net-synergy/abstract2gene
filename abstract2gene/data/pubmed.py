"""Download and manage data from Pubmed.

For attaching publication annotations to PubNet objects.
"""

__all__ = [
    "PubmedDownloader",
    "add_gene_edges",
]

import os

import numpy as np
import pandas as pd
from pubnet import PubNet

from ._downloader import FtpDownloader
from ._utils import default_cache_dir

_NAME = "pubmed"
_FILES = ["gene2pubmed.gz", "gene_info.gz"]


class PubmedDownloader(FtpDownloader):
    """Download gene files from Pubmed FTP server."""

    def __init__(self, cache_dir: str | None = None):
        super().__init__(_FILES, cache_dir)

    @property
    def name(self) -> str:
        return _NAME

    @property
    def server(self) -> str:
        return "ftp.ncbi.nlm.nih.gov"

    @property
    def subdir(self) -> str:
        return "/gene/DATA/"


def add_gene_edges(
    net: PubNet,
    cache_dir: str | None = None,
    replace: bool = False,
) -> None:
    """Add edges from pubmed gene files to the provided PubNet.

    Parameters
    ----------
    net : PubNet
        The net to attach the gene nodes and edges to.
    cache_dir : str, optional
        Where to search for downloaded edge files (default's to
        `default_cache_dir()`).
    replace : bool, optional
        What to do if Gene--Publication nodes already exist in `net`. By
        default (false) raise an error. If replace is true, remove the old
        edges and nodes.

    """

    def _read_pubmed_data(cache_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        edge_path = os.path.join(cache_dir, _FILES[0])
        metadata = os.path.join(cache_dir, _FILES[1])
        table = pd.read_table(
            edge_path,
            header=0,
            names=["TaxID", "NCBIGeneID", "PMID"],
            usecols=["PMID", "NCBIGeneID", "TaxID"],
            memory_map=True,
        )

        metadata_table = pd.read_table(
            metadata,
            header=0,
            names=[
                "TaxID",
                "NCBIGeneID",
                "GeneSymbol",
                "LocusTag",
                "Synonyms",
                "dbXrefs",
                "Chromosome",
                "MapLocation",
                "Description",
                "GeneType",
                "NomenclatureSymbol",
                "NomenclatureFull",
                "NomenclatureStatus",
                "Other",
                "Modified",
                "FeatureType",
            ],
            usecols=[
                "TaxID",
                "NCBIGeneID",
                "GeneSymbol",
                "Synonyms",
                "dbXrefs",
                "Description",
                "GeneType",
            ],
            memory_map=True,
        )

        return (table, metadata_table)

    cache_dir = cache_dir or default_cache_dir()

    if "Gene-Publication" in net.edges:
        if replace:
            net.drop(nodes=("Gene",), edges=(("Gene", "Publication"),))
        else:
            raise RuntimeError("Gene--Publication edges already in net.")

    accepted_download = False
    for file in _FILES:
        path = os.path.join(os.path.join(cache_dir, _NAME), file)
        if not os.path.exists(path):
            msg = f"{file} has not been downloaded, download now? (Y/n)"
            if not accepted_download and input(msg).lower() == "n":
                raise RuntimeError("Need to download data files.")

            accepted_download = True
            downloader = PubmedDownloader(cache_dir)
            downloader.download()

    table, metadata = _read_pubmed_data(cache_dir)

    node_data = table[
        [col for col in table.columns if col not in ("PMID", "TaxID")]
    ]
    node_data = node_data.drop_duplicates(
        ignore_index=True, subset=["NCBIGeneID"]
    )

    node_data = node_data.join(
        metadata.set_index("NCBIGeneID"),
        on="NCBIGeneID",
        how="left",
        validate="1:1",
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
