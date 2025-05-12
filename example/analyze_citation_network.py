from functools import partial
from time import time

import datasets
import numpy as np
from tqdm import tqdm

import example._config as cfg
from example._logging import log, set_log

set_log("citation_network")

dataset = datasets.load_dataset(f"{cfg.hf_user}/pubtator3_abstracts")["train"]
citing_pubs = dataset.filter(
    lambda example: len(example["reference"]) > 0, num_proc=cfg.max_cpu
)

print("Counting")
t = time()
n_total = len(dataset)
n_molecular = sum(
    len(example["gene"]) > 0 for example in tqdm(dataset, desc="Counting Refs")
)
n_citing = len(citing_pubs)
n_citing_molecular = sum(
    (
        len(example["gene"]) > 0
        for example in tqdm(citing_pubs, desc="Counting Refs")
    )
)

log(f"Total publications: {n_total}")
log(f"Number of publications with gene data: {n_molecular}")
log(f"Proportion publications with gene data: {n_molecular / n_total}")
log(f"Number of publications with reference data: {n_citing}")
log(f"Proportion publications with reference data: {n_citing / n_total}")
log(f"Number of referencing pubs with gene data: {n_citing_molecular}")
log(
    f"Proportion referencing pubs with gene data: {n_citing_molecular / n_citing}"
)

print(time() - t)
print("Unique")
unique_refs = np.unique(
    [ref for example in citing_pubs for ref in example["reference"]]
    + [example["pmid"] for example in citing_pubs]
)
print(time() - t)

print("Organizing")
t = time()
dataset = dataset.with_format(
    "numpy", columns=["pmid"], output_all_columns=True
)
indices = np.arange(len(dataset))
pmids = dataset["pmid"]
sort_idx = np.argsort(pmids)

indices = indices[sort_idx]
pmids = pmids[sort_idx]


def is_molecular(pmid: int) -> bool | None:
    idx = np.searchsorted(pmids, pmid)
    if idx == len(pmids) or pmids[idx] != pmid:
        return None

    return len(dataset[int(indices[idx])]["gene"]) > 0


labels = {int(k): is_molecular(k) for k in tqdm(unique_refs, desc="Labeling")}

print(time() - t)
print("Counting again")
t = time()
molecular = True
behavioral = False
edge_type = {
    molecular: {molecular: 0, behavioral: 0},
    behavioral: {molecular: 0, behavioral: 0},
}
for pub in tqdm(citing_pubs, desc="Typing"):
    parent_label = labels[pub["pmid"]]
    for ref in pub["reference"]:
        ref_label = labels[ref]
        if ref_label is None:
            continue

        edge_type[parent_label][ref_label] += 1

log("Edge distribution:")
log(f"  Molecular->Molecular: {edge_type[molecular][molecular]}")
log(f"  Molecular->Behavioral: {edge_type[molecular][behavioral]}")
log(f"  Behavioral->Molecular: {edge_type[behavioral][molecular]}")
log(f"  Behavioral->Behavioral: {edge_type[behavioral][behavioral]}")
print("DONE")
