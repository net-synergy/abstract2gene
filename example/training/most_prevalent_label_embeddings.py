"""Label embedding similarity for most prevalent genes.

Uses similarity between embeddings of abstracts to cluster the publications.
The gene labels for each publication are used as the ground truth partition the
partition generated from a clustering algorithm is compared to.

This looks closer at the most prevalent genes to compare performance between
the most popular vs less common genes to see if this has an impact.
The expectation is less common genes will be used in a narrower scope of
publications and therefore the model embedding will be more similar.

Explicitly use a template size of 1 here to emphasize variance between
publications.
"""

import argparse
import os

import datasets
import igraph as ig
import jax
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import speakeasy2 as se2
from sentence_transformers import SentenceTransformer

import example._config as cfg
from abstract2gene.dataset import mutators

EXPERIMENT = "most_prevalent_label_embedding_similarity"
seed = cfg.seeds[EXPERIMENT]
FIGDIR = f"figures/{EXPERIMENT}"

FINE_TUNED = f"{cfg.hf_user}/{cfg.encoder["remote_name"]}"

if not os.path.exists(FIGDIR):
    os.makedirs(FIGDIR)

n_labels = 15
samples_per_label = 100
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_labels",
        default=n_labels,
        type=int,
        help="Number of genes to use.",
    )
    parser.add_argument(
        "--samples_per_label",
        default=samples_per_label,
        type=int,
        help="Samples to collect per gene.",
    )
    args = parser.parse_args()
    n_labels = args.n_labels
    samples_per_label = args.samples_per_label


def filter_kth_prevalant_genes(
    dataset: datasets.Dataset, k: int, n_genes: int, symbols: list[str]
) -> tuple[datasets.Dataset, list[str]]:
    counts = np.bincount(jax.tree.leaves(dataset["gene"]))
    if (k - 1) == 0:
        gene_ids = np.argsort(counts)[-n_genes:]
    else:
        gene_ids = np.argsort(
            counts,
        )[-(k * n_genes) : -((k - 1) * n_genes)]

    dataset = dataset.filter(
        lambda example: any(np.isin(example["gene"], gene_ids)),
        num_proc=10,
    )

    symbols = np.take(symbols, gene_ids).tolist()  # type: ignore[assignment]
    newids = dict(zip(gene_ids, list(range(n_labels))))

    dataset = dataset.map(
        lambda example: {
            "gene": [
                newids[label] for label in example["gene"] if label in newids
            ]
        }
    )

    return (dataset, symbols)


def correlate(features: np.ndarray) -> np.ndarray:
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return features @ features.T


def plot_heatmap(corrs, symbols, ground_truth, name):
    names = {1: "Most prevalent", 5: "Less prevalent"}
    labels = [
        chr(65 + i) + r". \emph{" + names[k] + "}" for i, k in enumerate(corrs)
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(cfg.fig_width, cfg.fig_width / 2.4),
        constrained_layout=True,
    )

    for label, ax, k in zip(labels, axes, corrs):
        # This is left over from before but still used to convert the
        # correlation matrix to an igraph for order_nodes. Could do this more
        # directly.
        graph = se2.knn_graph(corrs[k], k=50)
        ordering_gt = se2.order_nodes(graph, ground_truth)
        norm = colors.Normalize(vmin=0, vmax=1)

        im = ax.imshow(corrs[k][np.ix_(ordering_gt, ordering_gt)], norm=norm)

        tick_pos = np.take(ground_truth.membership, ordering_gt)
        tick_pos = [
            int(np.median(np.where(tick_pos == i)))
            for i in range(len(symbols[k]))
        ]

        ax.set_xticks(tick_pos, ["" for _ in symbols[k]])
        ax.set_yticks(tick_pos, symbols[k])
        ax.set_facecolor("none")
        ax.set_xlabel(label)

    fig.patch.set_facecolor("none")
    fig.colorbar(
        im,
        ax=axes,
        fraction=0.075,
        pad=0.02,
        shrink=1,
        panchor=(0.9, 0.2),
    )

    plt.savefig(os.path.join(FIGDIR, f"{name}.{cfg.figure_ext}"))


def plot_violin(corrs, symbols, ground_truth, name):
    membership = ground_truth.membership
    n_samples = list(corrs.values())[0].shape[0]

    names = {1: "Most prevalent", 5: "Less prevalent"}
    for k in corrs:
        corrs[k] = corrs[k] - corrs[k].mean(axis=(0, 1))
        corrs[k] = corrs[k] / corrs[k].std(axis=(0, 1))

    fr = [
        symbols[k][membership[i]]
        for j in range(n_samples)
        for i in range(n_samples)
        for _ in corrs
        if i != j
    ]

    to = [
        symbols[k][membership[j]]
        for j in range(n_samples)
        for i in range(n_samples)
        for _ in corrs
        if i != j
    ]

    score = [
        model_corr[i, j]
        for j in range(n_samples)
        for i in range(n_samples)
        for model_corr in corrs.values()
        if i != j
    ]

    model = [
        names[k] for _ in range(n_samples * (n_samples - 1)) for k in corrs
    ]
    same = [head == tail for head, tail in zip(fr, to)]

    df = pd.DataFrame(
        dict(
            zip(
                ["from", "to", "model", "same", "score"],
                [fr, to, model, same, score],
            )
        )
    )

    dodge = p9.position_dodge(width=0.9)
    p = (
        p9.ggplot(df, p9.aes(x="model", y="score", fill="same"))
        + p9.geom_violin(position=dodge)
        + p9.labs(x="Model", y="Correlation (Normalized)", fill="Same Gene")
        + p9.theme(
            text=p9.element_text(family=cfg.font_family, size=cfg.font_size),
        )
    )
    p.save(
        os.path.join(FIGDIR, f"{name}_violin.{cfg.figure_ext}"),
        width=cfg.fig_width,
        height=cfg.fig_height,
    )


## Load dataset
dataset = datasets.load_dataset(
    f"{cfg.hf_user}/pubtator3_abstracts", data_files=cfg.LABEL_SIMILARITY_FILES
)["train"]
dataset = mutators.translate_to_human_orthologs(dataset, cfg.max_cpu)
symbols = mutators.get_gene_symbols(dataset)

embed_ft = SentenceTransformer(FINE_TUNED)

corrs = {1: np.array([]), 5: np.array([])}
syms = {1: [""], 5: [""]}
for k in corrs:
    ds_k, syms[k] = filter_kth_prevalant_genes(
        dataset, k=k, n_genes=n_labels, symbols=symbols
    )

    ds_k = ds_k.map(
        lambda examples: {
            FINE_TUNED: [
                embed_ft.encode(title + "[SEP]" + abstract)
                for title, abstract in zip(
                    examples["title"], examples["abstract"]
                )
            ],
        },
        batched=True,
        batch_size=30,
        remove_columns=["abstract"],
        desc="Encoding",
    )

    # Remove samples with multiple labels to prevent examples belonging to
    # multiple clusters.
    clusters = np.fromiter(
        (
            (i, sample[0])
            for i, sample in enumerate(ds_k["gene"])
            if len(sample) == 1
        ),
        dtype=(int, 2),
    )
    samples, clusters = clusters[:, 0], clusters[:, 1]

    # Randomly downsample to reduce computation and get an equal number of
    # samples per gene.
    rng = np.random.default_rng(seed=seed)
    indices = np.zeros((n_labels, samples_per_label), dtype=int)
    for i in range(n_labels):
        indices[i, :] = rng.choice(
            np.arange(clusters.shape[0])[clusters == i],
            samples_per_label,
            replace=False,
        )
    indices = np.concat(indices, axis=0)

    samples = samples[indices]
    ground_truth = ig.clustering.Clustering(clusters[indices])

    ds_k = ds_k.select(samples).with_format("numpy", columns=[FINE_TUNED])

    corrs[k] = correlate(ds_k[FINE_TUNED])

plot_heatmap(
    corrs,
    syms,
    ground_truth,
    f"{list(corrs.keys())[0]}_vs_{list(corrs.keys())[1]}",
)

plot_violin(
    corrs,
    syms,
    ground_truth,
    f"{list(corrs.keys())[0]}_vs_{list(corrs.keys())[1]}",
)
