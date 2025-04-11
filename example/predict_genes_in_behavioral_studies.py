"""Determine if abstract2gene can identify gene info in behavioral studies.

A set of experiments to see if publications that have not been given gene
annotations by PubTator3 have meaningful genetic components in their abstracts
that abstract2gene is able to pick out.
"""

import os
import sys

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9

import abstract2gene as a2g
from example import config as cfg

seed = cfg.seeds["predict_genes_in_behavioral_studies"]
FIGDIR = "figures/behavioral_genes"

if not os.path.exists(FIGDIR):
    os.makedirs(FIGDIR)

dataset = datasets.load_dataset(f"{cfg.hf_user}/pubtator3_abstracts")["train"]
rng = np.random.default_rng(seed=seed)
parent_publications = [
    int(pub)
    for pub in rng.integers(0, len(dataset), 10000)
    if len(dataset[int(pub)]["reference"]) > 0
    and len(dataset[int(pub)]["gene"]) > 0
]

parent_pmids = dataset[parent_publications]["pmid"]
reference_lists = dataset[parent_publications]["reference"]

dataset = dataset.with_format(
    "numpy", columns=["pmid"], output_all_columns=True
)
pmids = dataset["pmid"]
idx = np.arange(len(pmids))
sort_idx = np.argsort(pmids)

idx = idx[sort_idx]
pmids = pmids[sort_idx]

indices = [
    [
        int(idx[np.searchsorted(pmids, ref)])
        for ref in ref_list
        if ref < len(pmids)
    ]
    for ref_list in reference_lists
]

behavioral_indices = [
    [
        i
        for i, ref in zip(ref_indices, ref_list)
        if ((dataset[i]["pmid"] == ref) and (len(dataset[i]["gene"]) == 0))
    ]
    for ref_indices, ref_list in zip(indices, reference_lists)
]

molecular_indices = [
    [
        i
        for i, ref in zip(ref_indices, ref_list)
        if ((dataset[i]["pmid"] == ref) and (len(dataset[i]["gene"]) > 0))
    ]
    for ref_indices, ref_list in zip(indices, reference_lists)
]

selected_references = [
    (
        int(parent),
        int(rng.choice(ref_behave_indices, 1)[0]),
        int(rng.choice(ref_molec_indices, 1)[0]),
    )
    for parent, ref_behave_indices, ref_molec_indices in zip(
        parent_publications, behavioral_indices, molecular_indices
    )
    if (len(ref_behave_indices) > 0) and (len(ref_molec_indices) > 0)
]

random_behavioral_study: list[int] = []
while len(random_behavioral_study) < len(selected_references):
    ref = int(rng.integers(0, len(dataset), 1)[0])
    if (ref not in random_behavioral_study) and (
        len(dataset[ref]["gene"]) == 0
    ):
        random_behavioral_study.append(ref)

inputs = [
    [
        dataset[ref]["title"] + "[SEP]" + dataset[ref]["abstract"]
        for ref in (parent, behave_citation, molec_citation, random)
    ]
    for random, (parent, behave_citation, molec_citation) in zip(
        random_behavioral_study, selected_references
    )
]

for name in [f"a2g_768dim_per_batch_{2**n}" for n in range(1, 7)]:
    model = a2g.model.load_from_disk(name)
    probabilities = [
        np.array(model.predict(abstracts)) for abstracts in inputs
    ]
    parent = np.hstack([[i, i, i] for i in range(len(inputs))])
    group = np.hstack(
        [
            ["behavioral_reference", "molecular_reference", "random"]
            for i in range(len(inputs))
        ]
    )
    corr = np.hstack(
        [
            [
                (ref_set[0, :] @ ref_set[1, :]) /
                # / (ref_set[0, :] @ ref_set[0, :]),
                (
                    np.linalg.norm(ref_set[0, :])
                    * np.linalg.norm(ref_set[1, :])
                ),
                (ref_set[0, :] @ ref_set[2, :]) /
                # / (ref_set[0, :] @ ref_set[0, :]),
                (
                    np.linalg.norm(ref_set[0, :])
                    * np.linalg.norm(ref_set[2, :])
                ),
                (ref_set[0, :] @ ref_set[3, :]) /
                # / (ref_set[0, :] @ ref_set[0, :]),
                (
                    np.linalg.norm(ref_set[0, :])
                    * np.linalg.norm(ref_set[3, :])
                ),
            ]
            for ref_set in probabilities
        ]
    )

    corrs = pd.DataFrame(
        {"parent": parent, "group": group, "correlation": corr}
    )

    summary = (
        corrs.drop(columns=["parent"]).groupby("group").agg(["mean", "std"])
    )
    summary.columns = [col for _, col in summary.columns]
    summary = summary.reset_index()
    p = (
        p9.ggplot(corrs, p9.aes(fill="group", x="correlation", color="group"))
        + p9.geom_vline(
            p9.aes(xintercept="mean", color="group"),
            data=summary,
            linetype="dashed",
        )
        + p9.geom_histogram(alpha=0.3, binwidth=0.025, position="dodge")
        + p9.geom_text(
            p9.aes(label="mean", x="mean+0.05", y=200),
            color="black",
            size=8,
            va="bottom",
            data=summary,
            format_string="{:0.2f}",
        )
        + p9.theme(
            text=p9.element_text(family=cfg.font_family, size=cfg.font_size),
        )
    )
    p.save(
        os.path.join(FIGDIR, f"correlation_{name}.{cfg.figure_ext}"),
        width=cfg.fig_width,
        height=cfg.fig_height,
    )

    selected_genes = [np.argmax(p_set[0]) for p_set in probabilities]
    selected_probabilities = np.vstack(
        [
            (
                p_set[0][gene],
                p_set[1][gene],
                p_set[2][gene],
                p_set[3][gene],
            )
            for gene, p_set in zip(selected_genes, probabilities)
        ]
    )

    prob_df = pd.DataFrame(
        {
            "parent_prob": selected_probabilities[:, 0].repeat(3),
            "reference_prob": selected_probabilities[:, 1:].reshape((-1,)),
            "group": (
                ["behavioral", "molecular", "random"]
                * selected_probabilities.shape[0]
            ),
        }
    )

    p = (
        p9.ggplot(
            prob_df, p9.aes(x="parent_prob", y="reference_prob", color="group")
        )
        + p9.geom_point()
        + p9.geom_smooth()
        + p9.theme(
            text=p9.element_text(family=cfg.font_family, size=cfg.font_size),
        )
    )
    p.save(
        os.path.join(
            FIGDIR, f"selected_gene_correlation_{name}.{cfg.figure_ext}"
        ),
        width=cfg.fig_width,
        height=cfg.fig_height,
    )
