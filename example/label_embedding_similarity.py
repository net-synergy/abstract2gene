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

import datasets
import igraph as ig
import jax
import matplotlib.pyplot as plt
import numpy as np
import speakeasy2 as se2
from sentence_transformers import SentenceTransformer

import abstract2gene as a2g
from abstract2gene.data import encoder_path, model_path
from abstract2gene.dataset import mutators
from example import config as cfg

SEED = 0
N_LABELS = 15
SAMPLES_PER_LABEL = 100
MODEL = model_path("abstract2gene")
ENCODER = encoder_path("pubmedncl-abstract2gene")


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
    newids = dict(zip(gene_ids, list(range(N_LABELS))))

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
    graph = se2.knn_graph(corr, k=50)
    ordering_gt = se2.order_nodes(graph, ground_truth)

    fig, ax = plt.subplots()
    ax.imshow(corr[np.ix_(ordering_gt, ordering_gt)])

    fig.suptitle(title)

    tick_pos = np.take(ground_truth.membership, ordering_gt)
    tick_pos = [
        int(np.median(np.where(tick_pos == i))) for i in range(N_LABELS)
    ]

    ax.set_xticks([])
    ax.set_yticks(tick_pos, symbols)

    plt.savefig(f"figures/label_similarities/{name}.svg")
    # plt.show()


## Load dataset
dataset = datasets.load_dataset(
    "dconnell/pubtator3_abstracts", data_files=cfg.LABEL_SIMILARITY_FILES
)["train"]
symbols = mutators.get_gene_symbols(dataset)
model = a2g.model.load_from_disk(MODEL)
model.eval()

embed_orig = SentenceTransformer("malteos/PubMedNCL")
embed_ft = SentenceTransformer(ENCODER)

for k in [1, 5]:
    ds_k, sym_k = filter_kth_prevalant_genes(
        dataset, k=k, n_genes=N_LABELS, symbols=symbols
    )

    ds_k = ds_k.map(
        lambda examples: {
            "original": [
                embed_orig.encode(title + "[SEP]" + abstract)
                for title, abstract in zip(
                    examples["title"], examples["abstract"]
                )
            ],
            "fine_tuned": [
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
    ).map(
        lambda examples: {
            "transformed": [
                model(embedding) for embedding in examples["fine_tuned"]
            ]
        },
        batched=True,
        batch_size=1000,
        desc="Transforming",
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
    rng = np.random.default_rng(seed=SEED)
    indices = np.zeros((N_LABELS, SAMPLES_PER_LABEL), dtype=int)
    for i in range(N_LABELS):
        indices[i, :] = rng.choice(
            np.arange(clusters.shape[0])[clusters == i],
            SAMPLES_PER_LABEL,
            replace=False,
        )
    indices = np.concat(indices, axis=0)

    samples = samples[indices]
    ground_truth = ig.clustering.Clustering(clusters[indices])

    ds_k = ds_k.select(samples).with_format(
        "numpy", columns=["original", "fine_tuned", "transformed"]
    )

    for feature in ["original", "fine_tuned", "transformed"]:
        corr = correlate(ds_k[feature])
        plot(
            corr,
            sym_k,
            ground_truth,
            f"{feature}_k{k}",
            f"Similarity of {k}{'st' if k == 1 else 'th'} most prevalent"
            + f" 15 genes with {feature} embeddings",
        )
