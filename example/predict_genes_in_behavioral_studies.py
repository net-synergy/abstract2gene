import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9

import abstract2gene as a2g

SEED = 10

dataset = datasets.load_dataset("dconnell/pubtator3_abstracts")["train"]
rng = np.random.default_rng(seed=SEED)
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
    [int(idx[np.searchsorted(pmids, ref)]) for ref in ref_list]
    for ref_list in reference_lists
]

indices = [
    [
        i
        for i, ref in zip(ref_indices, ref_list)
        if ((dataset[i]["pmid"] == ref) and (len(dataset[i]["gene"]) == 0))
    ]
    for ref_indices, ref_list in zip(indices, reference_lists)
]

selected_references = [
    (
        int(parent),
        int(rng.choice(ref_indices, 1)[0]),
    )
    for parent, ref_indices in zip(parent_publications, indices)
    if len(ref_indices) > 0
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
        for ref in (parent, citation, random)
    ]
    for random, (parent, citation) in zip(
        random_behavioral_study, selected_references
    )
]

for name in [f"a2g_768dim_per_batch_{2**n}" for n in range(1, 7)]:
    model = a2g.model.load_from_disk(name)
    probabilities = [
        np.array(model.predict(abstracts)) for abstracts in inputs
    ]
    parent = np.hstack([[i, i] for i in range(len(inputs))])
    group = np.hstack([["reference", "random"] for i in range(len(inputs))])
    corr = np.hstack(
        [
            [
                (ref_set[0, :] @ ref_set[1, :])
                / (
                    np.linalg.norm(ref_set[0, :])
                    * np.linalg.norm(ref_set[1, :])
                ),
                (ref_set[0, :] @ ref_set[2, :])
                / (
                    np.linalg.norm(ref_set[0, :])
                    * np.linalg.norm(ref_set[2, :])
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
    )
    p.save(f"figures/behavioral_genes/correlation_{name}.png", dpi=600)

    selected_genes = [np.argmax(p_set[0]) for p_set in probabilities]
    selected_probabilities = np.vstack(
        [
            (p_set[0][gene], p_set[1][gene], p_set[2][gene])
            for gene, p_set in zip(selected_genes, probabilities)
        ]
    )

    fig, ax = plt.subplots()
    sort_idx = np.argsort(selected_probabilities[:, 0])
    ax.plot(
        selected_probabilities[sort_idx, 0],
        selected_probabilities[sort_idx, 1],
        label="citation",
    )
    ax.plot(
        selected_probabilities[sort_idx, 0],
        selected_probabilities[sort_idx, 2],
        label="random",
    )
    ax.legend(loc="upper left")
    ax.set_xlabel("parent prediction")
    ax.set_ylabel("reference prediction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.savefig(
        f"figures/behavioral_genes/selected_gene_correlation_{name}.png",
        dpi=600,
    )
    plt.close()
