"""A set of functions for modifying datasets.

Some of the functions are intended to use with map to add data to a dataset,
others are used to obtain separate objects for working with the datasets.

Since these collect data from foreign sources, many of the functions depend on
downloading large files.
"""

__all__ = ["attach_pubmed_genes", "get_gene_symbols"]

from typing import Any

import datasets
import numpy as np
import pandas as pd

from abstract2gene.data import PubmedDownloader


def attach_pubmed_genes(
    dataset: datasets.Dataset,
    name: str,
    max_cpu: int = 1,
    batch_size: int = 1000,
) -> datasets.Dataset:
    """Attach gene labels from the gene2pubmed table.

    This function should be called with a full dataset rather than with map.
    """

    def read_pubmed_data() -> pd.DataFrame:
        pubmed_downloader = PubmedDownloader()
        pubmed_downloader.files = ["gene2pubmed.gz"]
        files = pubmed_downloader.download()

        return pd.read_table(
            files[0],
            header=0,
            names=["TaxID", "GeneID", "PMID"],
            usecols=["PMID", "GeneID"],
        ).sort_values("PMID")

    def add_genes(
        examples: dict[str, Any],
        name: str,
        edges: pd.DataFrame,
        id2idx,
        labels,
    ) -> dict[str, Any]:

        def get_genes(pmid: int) -> list[str]:
            start = np.searchsorted(edges["PMID"], pmid)
            end = start
            while (end < edges.shape[0]) and (edges["PMID"].iloc[end] == pmid):
                end += 1

            return [
                id2idx(str(gene_id))
                for gene_id in edges["GeneID"].iloc[start:end]
                if str(gene_id) in labels
            ]

        return {name: [get_genes(pmid) for pmid in examples["pmid"]]}

    pubmed_edges = read_pubmed_data()
    if isinstance(dataset, datasets.dataset_dict.DatasetDict):
        key = list(dataset.keys())[0]
        features = dataset[key].features.copy()
    else:
        features = dataset.features.copy()

    id2idx = features["gene"].feature.str2int
    labels = features["gene"].feature.names
    features[name] = features["gene"]
    return dataset.map(
        add_genes,
        batched=True,
        batch_size=batch_size,
        features=features,
        fn_kwargs={
            "edges": pubmed_edges,
            "name": name,
            "id2idx": id2idx,
            "labels": labels,
        },
        num_proc=max_cpu,
        desc="Attach pubmed genes",
    )


def get_gene_symbols(dataset: datasets.Dataset) -> list[str]:
    """Grab gene symbols from PubMed's gene_info file.

    Returns a list matching the ith symbol to gene label i.
    """

    def read_symbol_table() -> pd.DataFrame:
        pubmed_downloader = PubmedDownloader()
        pubmed_downloader.files = ["gene_info.gz"]
        files = pubmed_downloader.download()

        return pd.read_table(
            files[0], usecols=["GeneID", "Symbol"]
        ).sort_values("GeneID")

    if isinstance(dataset, datasets.dataset_dict.DatasetDict):
        key = list(dataset.keys())[0]
        features = dataset[key].features.copy()
    else:
        features = dataset.features.copy()

    gene_ids = [int(gid) for gid in features["gene"].feature.names]
    symbol_table = read_symbol_table()
    symbol_table = symbol_table.merge(
        pd.DataFrame({"GeneID": gene_ids}), how="right", on="GeneID"
    )
    symbol_table["Symbol"] = symbol_table["Symbol"].fillna(
        symbol_table["GeneID"].transform(str)
    )

    return symbol_table["Symbol"].array.tolist()
