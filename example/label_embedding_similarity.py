"""Check the within label abstract similarity.

Uses similarity between embeddings of abstracts to cluster the publications.
The gene labels for each publication are used as the ground truth partition the
partition generated from a clustering algorithm is compared to.

Publications with multiple labels in the set of labels are dropped to prevent
overlapping community structure.

If the embeddings do a good job of pulling attention to the relevant aspects of
an abstract needed to classify the abstract, we expect clear strong clusters
for each gene. Otherwise, abstracts for a given label will be fragmented across
multiple clusters. If particularly bad, there will be mixed clusters.

If there are multiple clusters per label, a single template per gene may not be
adequate.

We can use this example to compare embeddings produced by multiple models.
"""

import argparse
import os
import sys

import datasets
import igraph as ig
import jax
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import speakeasy2 as se2
from sentence_transformers import SentenceTransformer

from abstract2gene.data import encoder_path
from abstract2gene.dataset import mutators
from example import config as cfg

seed = cfg.seeds["label_embedding_similarity"]
FIGDIR = "figures/label_similarities"
ENCODER = encoder_path("PubMedNCL-abstract2gene")

if not os.path.exists(FIGDIR):
    os.makedirs(FIGDIR)

n_labels = 15
samples_per_label = 100
if __name__ == "__main__" and len(sys.argv) == 2:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "n_labels",
        required=False,
        default=n_labels,
        help="Number of genes to use.",
    )
    parser.add_argument(
        "samples_per_label",
        required=False,
        default=samples_per_label,
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


def plot(corr, symbols, ground_truth, name, title):
    # Use a different width for square figures than the global config width.
    fig_width = 0.7 * cfg.text_width

    # This is left over from before but still used to convert the correlation
    # matrix to an igraph for order_nodes. Could do this more directly.
    graph = se2.knn_graph(corr, k=50)
    ordering_gt = se2.order_nodes(graph, ground_truth)
    norm = colors.Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(fig_width, fig_width))
    im = ax.imshow(corr[np.ix_(ordering_gt, ordering_gt)], norm=norm)

    tick_pos = np.take(ground_truth.membership, ordering_gt)
    tick_pos = [
        int(np.median(np.where(tick_pos == i))) for i in range(n_labels)
    ]

    ax.set_xticks(
        tick_pos, symbols, rotation=45, ha="right", rotation_mode="anchor"
    )
    ax.set_yticks(tick_pos, symbols)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    plt.tight_layout(pad=0, rect=(0, 0, 1, 0.96))

    plt.savefig(os.path.join(FIGDIR, f"{name}.{cfg.figure_ext}"))


## Load dataset
dataset = datasets.load_dataset(
    f"{cfg.hf_user}/pubtator3_abstracts", data_files=cfg.LABEL_SIMILARITY_FILES
)["train"]
symbols = mutators.get_gene_symbols(dataset)

scibert = SentenceTransformer(cfg.MODELS["scibert"])
embed_orig = SentenceTransformer(cfg.MODELS["PubMedNCL"])
embed_ft = SentenceTransformer(ENCODER)

for k in [1, 5]:
    ds_k, sym_k = filter_kth_prevalant_genes(
        dataset, k=k, n_genes=n_labels, symbols=symbols
    )

    ds_k = ds_k.map(
        lambda examples: {
            "scibert": [
                scibert.encode(title + "[SEP]" + abstract)
                for title, abstract in zip(
                    examples["title"], examples["abstract"]
                )
            ],
            "pubmed-ncl": [
                embed_orig.encode(title + "[SEP]" + abstract)
                for title, abstract in zip(
                    examples["title"], examples["abstract"]
                )
            ],
            "fine-tuned": [
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

    ds_k = ds_k.select(samples).with_format(
        "numpy", columns=["pubmed-ncl", "fine-tuned", "scibert"]
    )

    for feature in ["pubmed-ncl", "fine-tuned", "scibert"]:
        corr = correlate(ds_k[feature])
        plot(
            corr,
            sym_k,
            ground_truth,
            f"{feature}_k{k}",
            f"Similarity of {k}{'st' if k == 1 else 'th'} most prevalent"
            + f" 15 genes with {feature} embeddings",
        )
