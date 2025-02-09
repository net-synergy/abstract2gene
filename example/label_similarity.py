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

import os

import igraph as ig
import jax
import matplotlib.pyplot as plt
import numpy as np
import speakeasy2 as se2
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

from abstract2gene.data import dataset_path, model_path
from abstract2gene.dataset import mutators

DATASET = "bioc_small_embeddings"
DATASET = "bioc_small"
N_LABELS = 15
SAMPLES_PER_LABEL = 100
LABEL_SET = "gene2pubtator"
MODEL = model_path("specter-abstract-genes")
name = "specter-finetuned_and_trained"


## Load data
dataset = load_from_disk(dataset_path(DATASET))
symbols = mutators.get_gene_symbols(dataset)

embed = SentenceTransformer(MODEL)
## Filter to most common genes
counts = np.bincount(jax.tree.leaves(dataset[LABEL_SET]))
gene_ids = np.argsort(
    counts,
)[-N_LABELS:]

dataset = dataset.filter(
    lambda example: any(np.isin(example["gene2pubtator"], gene_ids)),
    num_proc=10,
).map(
    lambda examples: {
        "embedding": [
            embed.encode(abstract) for abstract in examples["abstract"]
        ]
    },
    batched=True,
    batch_size=10,
)

symbols = np.take(symbols, gene_ids)
newids = dict(zip(gene_ids, list(range(N_LABELS))))
labels = [
    [newids[label] for label in sample if label in newids]
    for sample in dataset["gene2pubtator"]
]

# Remove samples with multiple labels to prevent examples beloning to multiple
# clusters.
clusters = np.fromiter(
    ((i, sample[0]) for i, sample in enumerate(labels) if len(sample) == 1),
    dtype=(int, 2),
)
samples, clusters = clusters[:, 0], clusters[:, 1]

# Randomly downsample to reduce computation and get an equal number of samples
# per gene.
rng = np.random.default_rng(seed=0)
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

dataset = dataset.select(samples).with_format("numpy", columns="embedding")

## Cluster cosine similarity between embeddings
# Uses KNN-graph to reduce total edges.
embeddings = dataset["embedding"]
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
corr = embeddings @ embeddings.T
graph = se2.knn_graph(corr, k=10)
partition = se2.cluster(graph)

## Compare to ground truth and visualize
ig.compare_communities(partition, ground_truth, "NMI")

ordering_gt = se2.order_nodes(graph, ground_truth)
ordering_embeddings = se2.order_nodes(graph, partition)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(corr[np.ix_(ordering_gt, ordering_gt)])
axes[1].imshow(corr[np.ix_(ordering_embeddings, ordering_embeddings)])

axes[0].set_title("Ground truth ordering")
axes[1].set_title("Ordering from embedding similarity")
fig.suptitle(
    "Model transformed embedding similarity for abstracts"
    + f"\n(filtered to {N_LABELS} most prevalent genes)"
)

tick_pos = np.take(ground_truth.membership, ordering_gt)
tick_pos = [int(np.median(np.where(tick_pos == i))) for i in range(N_LABELS)]

axes[0].set_xticks([])
axes[1].set_xticks([])
axes[0].set_yticks(tick_pos, symbols)
axes[1].set_yticks([])

plt.savefig("figures/embedding_similarities.svg")
plt.show()
