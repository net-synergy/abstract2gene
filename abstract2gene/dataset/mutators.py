"""A set of functions for modifying datasets.

Some of the functions are intended to use with map to add data to a dataset,
others are used to obtain separate objects for working with the datasets.

Since these collect data from foreign sources, many of the functions depend on
downloading large files.
"""

__all__ = [
    "attach_pubmed_genes",
    "get_gene_symbols",
    "translate_to_human_orthologs",
    "mask_abstract",
    "attach_references",
    "augment_labels",
]

import json
import os
from collections import defaultdict
from typing import Any, Sequence

import datasets
import numpy as np
import pandas as pd
import pubmedparser
import pubmedparser.ftp
from tqdm import tqdm

from abstract2gene.data import PubmedDownloader, default_cache_dir

from ._utils import lol_to_csc


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


def translate_to_human_orthologs(
    dataset: datasets.Dataset,
    max_cpu: int = 1,
) -> datasets.Dataset:
    """Convert all genes to human orthologs.

    For any gene IDs that are not associated with the human Tax ID (9606), look
    for an ortholog and switch it out for the human gene. If no ortholog is
    found, remove the label.
    """
    human_tax_id = 9606

    def read_orthologs() -> pd.DataFrame:
        pubmed_downloader = PubmedDownloader()
        pubmed_downloader.files = ["gene_orthologs.gz"]
        files = pubmed_downloader.download()

        df = pd.read_table(
            files[0],
            header=0,
            names=[
                "TaxID",
                "GeneID",
                "Relationship",
                "Other_TaxID",
                "Other_GeneID",
            ],
            usecols=["TaxID", "GeneID", "Other_TaxID", "Other_GeneID"],
        )

        human_genes = pd.concat(
            (
                df["GeneID"][df["TaxID"] == human_tax_id],
                df["Other_GeneID"][df["Other_TaxID"] == human_tax_id],
            )
        )

        other_genes = pd.concat(
            (
                df["Other_GeneID"][df["TaxID"] == human_tax_id],
                df["GeneID"][df["Other_TaxID"] == human_tax_id],
            )
        )

        gene_map = pd.DataFrame(
            {"HumanGeneID": human_genes, "OtherGeneID": other_genes}
        ).sort_values("OtherGeneID")

        human_genes.sort_values().drop_duplicates()

        return (human_genes, gene_map)

    def is_human(gene_id: int | np.int_) -> np.bool_:
        idx = np.searchsorted(human_genes, gene_id)

        return (idx < len(human_genes)) and (human_genes.iloc[idx] == gene_id)

    def retrieve_human_ortholog(gene_id: int | np.int_) -> int | None:
        if is_human(gene_id):
            return int(gene_id)

        idx = np.searchsorted(gene_map["OtherGeneID"], gene_id)

        if gene_map["OtherGeneID"].iloc[idx] == gene_id:
            return gene_map["HumanGeneID"].iloc[idx]

        return None

    def convert_genes(example: dict[str, Any]):
        return {
            "gene": [
                gene2idx(gene)
                for gene in (
                    retrieve_human_ortholog(idx2gene(gene_id))
                    for gene_id in example["gene"]
                )
                if gene is not None
            ]
        }

    def idx2gene(gene_id: int) -> int:
        return old_genes[gene_id]

    def gene2idx(gene: int) -> np.int_:
        return np.searchsorted(new_genes, gene)

    human_genes, gene_map = read_orthologs()
    features = dataset.features.copy()
    old_genes = [int(gene) for gene in features["gene"].feature.names]

    new_genes = [
        gene
        for gene in {
            retrieve_human_ortholog(gene)
            for gene in tqdm(old_genes, desc="Reindexing genes")
        }
        if gene
    ]
    new_genes.sort()

    features["gene"] = datasets.Sequence(
        datasets.ClassLabel(names=[str(gene) for gene in new_genes])
    )

    return dataset.map(
        convert_genes,
        features=features,
        batched=False,
        num_proc=max_cpu,
        desc="Converting to human genes",
    )


def mask_abstract(
    dataset: datasets.Dataset,
    ann_type: str | Sequence[str],
    permute_prob: float = 0.0,
    seed: int = 0,
    mask_token: str = "[MASK]",
    max_cpu: int = 1,
) -> datasets.Dataset:
    """Replace annotations in abstracts with mask token."""

    def token_selector_generator(n_picks):
        if permute_prob <= 0:
            return lambda _: mask_token

        rng = np.random.default_rng(seed)
        count_pick = 0
        picks = rng.random((n_picks,)) > permute_prob
        n_pubs = len(dataset)
        count_pub = 0
        pubs = rng.integers(0, n_pubs, (n_picks,))

        def get_token(ann_types: Sequence[str]) -> str:
            nonlocal count_pick, picks, count_pub, pubs, rng

            if count_pick == n_picks:
                count_pick = 0
                picks = rng.random((n_picks,)) > permute_prob

            count_pick += 1
            if picks[count_pick - 1]:
                return mask_token

            n_anns = 0
            while n_anns == 0:
                if count_pub == n_picks:
                    count_pub = 0
                    pubs = rng.integers(0, n_pubs, (n_picks,))

                publication = dataset[int(pubs[count_pub])]
                count_pub += 1
                n_anns = len(
                    [
                        ann
                        for ann_type in ann_types
                        for ann in publication[ann_type]
                    ]
                )

            choice = int(rng.integers(0, n_anns, (1,)))
            anns = [
                ann
                for ann in publication["annotation"]
                if ann["type"].lower() in ann_types
            ]
            ann = anns[choice]

            pos_start = ann["offset"]
            pos_end = pos_start + ann["length"]
            return publication["abstract"][pos_start:pos_end]

        return get_token

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
                    (letter if mask[i] > 0 else get_token(ann_types))
                    for i, letter in enumerate(abstract)
                    if mask[i]
                )
            )
        }

    if isinstance(ann_type, str):
        ann_type = [ann_type]

    ann_type = [t.lower() for t in ann_type]
    get_token = token_selector_generator(10_000)

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


def _calculate_augmented_labels(label: str, fpath: str):
    import jax
    from scipy.stats import binom

    def sorted_isin(el, arr) -> np.bool_:
        idx = np.searchsorted(arr, el)
        return arr[idx] == el

    def build_behavioral_mask(
        references: list[list[int]],
    ) -> list[list[np.bool_]]:
        refs, treedef = jax.tree.flatten(references)
        mask = np.isin(refs, unlabeled_pmids)

        return jax.tree.unflatten(treedef, mask)

    def filter_to_behavioral(
        example: dict[str, Any],
        index: int,
        behavioral_mask: list[list[np.bool_]],
    ) -> dict[str, Any]:
        idx = np.arange(len(example["reference"]))[behavioral_mask[index]]

        return {"reference": np.take(example["reference"], idx)}

    dataset = datasets.load_dataset("dconnell/pubtator3_abstracts")["train"]
    annotated_pubs = dataset.filter(
        lambda example: len(example["reference"]) > 0
        and len(example[label]) > 0,
        num_proc=60,
    )

    # Restructuring dataset is slow when large. Saves time by doing it once.
    references = annotated_pubs["reference"]

    uniq_refs = np.unique_values(
        [ref for ref_list in references for ref in ref_list]
    )

    unlabeled_pubs = dataset.filter(
        lambda example: len(example[label]) == 0
        and sorted_isin(example["pmid"], uniq_refs),
        num_proc=60,
    )

    unlabeled_pmids = np.unique_values(unlabeled_pubs["pmid"])
    unlabeled_mask = build_behavioral_mask(references)

    annotated_pubs = annotated_pubs.map(
        filter_to_behavioral,
        with_indices=True,
        fn_kwargs={"behavioral_mask": unlabeled_mask},
    ).filter(lambda example: len(example["reference"]) > 0)

    cited_by = lol_to_csc(annotated_pubs["reference"])
    cited_by = cited_by[:, unlabeled_pmids]

    mask = cited_by.sum(axis=0) > 2
    unlabeled_pmids = np.take(
        unlabeled_pmids, np.arange(cited_by.shape[1])[mask]
    )
    cited_by = cited_by[:, mask]

    mask = cited_by.sum(axis=1) > 0
    cited_by = cited_by[mask, :]
    cited_by = cited_by.astype(int)

    if label == "gene":
        annotated_pubs = translate_to_human_orthologs(
            annotated_pubs, max_cpu=60
        )

    annotations = lol_to_csc(annotated_pubs["gene"])
    annotations = annotations[mask, :]
    annotations = annotations.astype(int)

    probs = annotations.mean(axis=0)
    events = cited_by.T @ annotations
    n = cited_by.sum(axis=0)
    thresh = 0.05 / np.prod(events.shape)

    augmented_anns: dict[int, list[int]] = defaultdict(list)
    for ann in range(events.shape[1]):
        pubs, _ = events[:, [ann]].nonzero()

        # -1 because we want probability of observing at least this many events
        # as opposed to more than this many events.
        unexpected = (
            binom.sf(events[pubs, ann].toarray() - 1, n[pubs], probs[ann])
            < thresh
        )

        for pub_idx in pubs[unexpected]:
            augmented_anns[int(unlabeled_pmids[pub_idx])].append(ann)

    with open(fpath, "w") as js:
        json.dump(augmented_anns, js)


def augment_labels(
    dataset: datasets.Dataset, label: str, prop: float, seed: int
):
    """Augment a dataset with inferred labels.

    Uses the citation network to infer annotations for publications that were
    not given any.

    Attempts to enrich unlabeled publication so that the final proportion of
    labeled publications is `prop` augmented and `1 - prop` PubTator3 labels.
    """

    def searchsorted(arr, val) -> int:
        idx = np.searchsorted(arr, val)
        if idx == len(arr) or arr[idx] != val:
            return -1

        return int(idx)

    rng = np.random.default_rng(seed)
    fpath = os.path.join(default_cache_dir(), f"{label}_augmented_labels.json")
    if not os.path.exists(fpath):
        print(
            "Augmented labels not cached; calculating. This may take a while."
        )
        _calculate_augmented_labels(label, fpath)

    with open(fpath, "r") as js:
        augmented_labels = json.load(js)

    annotated_pubs = dataset.filter(lambda example: len(example[label]) > 0)
    n_annotated = len(annotated_pubs)
    n_augmented = int((n_annotated * prop) / (1 - prop))

    mixin_pmids = rng.permuted(list(augmented_labels.keys()))
    ds_pmids = np.asarray(dataset["pmid"])
    indices = np.arange(len(ds_pmids))
    shuff = np.argsort(ds_pmids)
    indices = indices[shuff]
    ds_pmids = ds_pmids[shuff]

    count = 0
    for pmid in mixin_pmids:
        if count >= n_augmented:
            break

        idx = searchsorted(ds_pmids, int(pmid))
        if idx > 0:
            count += 1
            dataset[idx][label] = augmented_labels[pmid]

    if count < n_augmented:
        print(
            "Did not find enough augmented labels. Dataset enriched with "
            + f"{100 * (count / (count + n_annotated))}% augmented labels."
        )

    return dataset
