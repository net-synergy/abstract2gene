"""Download and manage data from pubtator (or pubmed)."""

import datetime
import os
from ftplib import FTP
from typing import Iterator

import numpy as np
import pandas as pd
from pubnet import PubNet

from ._utils import default_cache_dir

__ALL__ = ["download_gene_edges", "gene_pmid_edges"]

ftp_info = {
    "pubtator": {
        "server": "ftp.ncbi.nlm.nih.gov",
        "dir": "pub/lu/PubTator3",
        "files": ["gene2pubtator3.gz"],
    },
    "pubmed": {
        "server": "ftp.ncbi.nlm.nih.gov",
        "dir": "/gene/DATA/",
        "files": ["gene2pubmed.gz", "gene_info.gz"],
    },
}


def download_gene_edges(place: str, cache_dir: str | None = None) -> None:
    """Download pubtator's gene -> PMID annotations.

    Parameters
    ----------
    place : str, {"pubtator", "pubmed"}
        Which site to download edges from.
    cache_dir : optional str
        Where to store the file (defaults to `default_cache_dir`).

    See Also
    --------
    `abstract2gene.data.download` for a central download API for the package.
    `abstract2gene.data.default_cache_dir`

    """

    def _remote_modification(
        file: str,
        ls: Iterator[tuple[str, dict[str, str]]],
    ) -> datetime.datetime:
        mod_string = [f[1]["modify"] for f in ls if f[0] == file][0]
        mod_string = mod_string[:8] + "T" + mod_string[8:]
        return datetime.datetime.fromisoformat(mod_string)

    def _is_old(local: str, remote_modtime: datetime.datetime) -> bool:
        local_modtime = datetime.datetime.fromtimestamp(
            os.stat(local).st_mtime
        )

        return local_modtime < remote_modtime

    def _download_file(file: str):
        local_file = os.path.join(cache_dir or default_cache_dir(), file)

        if os.path.exists(local_file):
            remote_update = _remote_modification(
                file, ftp.mlsd(facts=["modify"])
            )
            if _is_old(local_file, remote_update):
                os.unlink(local_file)
            else:
                print("Current gene file already downloaded.")
                return

        print(f"Starting download to {local_file}")
        with open(local_file, "wb") as fp:
            ftp.retrbinary(f"RETR {file}", fp.write)

    ftp = FTP(ftp_info[place]["server"])
    ftp.login()
    ftp.cwd(ftp_info[place]["dir"])

    for file in ftp_info[place]["files"]:
        _download_file(file)

    ftp.quit()


def add_gene_edges(
    place: str,
    net: PubNet,
    cache_dir: str | None = None,
    replace: bool = False,
) -> None:
    """Add edges from a gene file to the provided PubNet.

    Parameters
    ----------
    place : str, {"pubtator", "pubmed"}
        Which upstream data provider's gene edges to use.
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

    def _read_pubtator_data(cache_dir: str) -> pd.DataFrame:
        edge_path = os.path.join(cache_dir, ftp_info["pubtator"]["files"][0])
        table = pd.read_table(
            edge_path,
            header=None,
            names=["PMID", "Type", "NCBIGeneID", "GeneSymbol", "Resource"],
            usecols=["PMID", "NCBIGeneID", "GeneSymbol"],
            memory_map=True,
        )
        table["NCBIGeneID"] = (
            table["NCBIGeneID"]
            .astype("str")
            .map(lambda x: x.split(";")[0])
            .astype("int64")
        )

        return table

    def _read_pubmed_data(cache_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        edge_path = os.path.join(cache_dir, ftp_info["pubmed"]["files"][0])
        metadata = os.path.join(cache_dir, ftp_info["pubmed"]["files"][1])
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

    for file in ftp_info[place]["files"]:
        path = os.path.join(cache_dir, file)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{file} not found. Download with "download_gene_edges".'
            )

    if place == "pubtator":
        table = _read_pubtator_data(cache_dir)
    else:
        table, metadata = _read_pubmed_data(cache_dir)

    node_data = table[
        [col for col in table.columns if col not in ("PMID", "TaxID")]
    ]
    node_data = node_data.drop_duplicates(
        ignore_index=True, subset=["NCBIGeneID"]
    )

    if place == "pubmed":
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
