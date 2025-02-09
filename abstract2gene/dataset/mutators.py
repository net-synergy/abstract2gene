"""A set of functions for modifying datasets.

Some of the functions are intended to use with map to add data to a dataset,
others are used to obtain separate objects for working with the datasets.

Since these collect data from foreign sources, many of the functions depend on
downloading large files.
"""

__all__ = [
    "attach_pubmed_genes",
    "get_gene_symbols",
    "mask_abstract",
    "attach_references",
]

import os
from typing import Any, Sequence

import datasets
import numpy as np
import pandas as pd
import pubmedparser
import pubmedparser.ftp

from abstract2gene.data import PubmedDownloader, default_cache_dir


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


def mask_abstract(
    dataset: datasets.Dataset,
    ann_type: str | Sequence[str],
    mask_token: str = "[MASK]",
    max_cpu: int = 1,
) -> datasets.Dataset:
    """Replace annotations in abstracts with mask token."""

    def mask_example(
        example: dict[str, list[Any]],
        ann_types: Sequence[str],
        mask_token: str,
    ):
        abstract = example["abstract"]
        mask = np.ones((len(abstract),))
        for ann in example["annotation"]:
            if ann["type"].lower() not in ann_types:
                continue

            pos_start = ann["offset"]
            pos_end = pos_start + ann["length"]
            mask[pos_start + 1 : pos_end] = 0
            mask[pos_start] = -1

        return {
            "abstract": "".join(
                (
                    letter if mask[i] > 0 else mask_token
                    for i, letter in enumerate(abstract)
                    if mask[i]
                )
            )
        }

    if isinstance(ann_type, str):
        ann_type = [ann_type]

    ann_type = [t.lower() for t in ann_type]
    return dataset.map(
        mask_example,
        batched=False,
        num_proc=max_cpu,
        fn_kwargs={"ann_types": ann_type, "mask_token": mask_token},
    )


def attach_references(
    dataset: datasets.Dataset, max_cpu: int = 1, batch_size: int = 1000
) -> datasets.Dataset:
    """Add a reference feature to the dataset.

    Uses the references found in PubMed.

    Note: This will download a lot of data.
    """

    def add_references(
        examples: dict[str, Any], table: pd.DataFrame
    ) -> dict[str, Any]:
        def get_references(pmid: int) -> list[str]:
            start = np.searchsorted(table["from"], pmid)
            end = start
            while (end < table.shape[0]) and (table["from"].iloc[end] == pmid):
                end += 1

            return list(table["to"].iloc[start:end])

        return {
            "reference": [get_references(pmid) for pmid in examples["pmid"]]
        }

    files = pubmedparser.ftp.download("all")
    reference_path = "/".join(
        [
            "/PubmedArticle",
            "PubmedData",
            "ReferenceList",
            "Reference",
            "ArticleIdList",
            "ArticleId",
            "[@IdType='pubmed']",
        ]
    )
    structure = {
        "root": "//PubmedArticleSet",
        "key": "/PubmedArticle/MedlineCitation/PMID",
        "Reference": reference_path,
    }
    data_dir = default_cache_dir("pubmedparser")

    # Files that pubmedparser says are malformed.
    bad_files = ["pubmed25n0953.xml.gz"]
    files = [f for f in files if os.path.basename(f) not in bad_files]

    print("Parsing XML files...")
    results = pubmedparser.read_xml(files, structure, data_dir, n_threads=32)

    print("Loading table...")
    table = pd.read_table(
        os.path.join(results, "Reference.tsv"),
        sep="\t",
        header=None,
        skiprows=1,
        names=["from", "to"],
        dtype={"from": int, "to": str},
    ).sort_values("from")

    # There's a small number of non-PMID values that get in there.
    table = table[table["to"].str.contains(r"^\d*$")]
    table["to"] = table["to"].transform(int)

    return dataset.map(
        add_references,
        batched=True,
        batch_size=batch_size,
        num_proc=max_cpu,
        fn_kwargs={"table": table},
        desc="Attaching references",
    )
