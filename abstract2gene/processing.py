import jax.numpy as jnp
import numpy as np
from pubnet import PubNet

__ALL__ = ["net2dataset"]


def net2dataset(
    net: PubNet,
    features: str = "Abstract_embedding",
    labels: str = "Gene",
    min_label_occurrences: int = 50,
    remove_baseline: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the network as a matrix of features and a matrix of labels.

    Features are normalized by publication so each publication's features has
    an l2 norm of 1.

    Features are samples (publications) x features and labels are samples x
    labels (binary vector).

    Label IDs is an array of the label nodes index that are in used in the
    labels. Labels are limited to genes with at least `min_label_occurrences`.

    If `remove_baseline` subtract the average of each feature.
    """
    embeddings_edge = net.get_edge(features, "Publication")
    n_features = np.sum(embeddings_edge["Publication"] == 0)
    embeddings = embeddings_edge.feature_vector("embedding").reshape(
        (-1, n_features)
    )

    if remove_baseline:
        baseline = embeddings.mean(axis=0, keepdims=True)
        embeddings = embeddings - baseline

    embeddings = embeddings / np.reshape(
        np.linalg.norm(embeddings, axis=1), shape=(-1, 1)
    )

    label_edges = net.get_edge("Publication", labels)
    label_frequencies = np.unique_counts(label_edges[labels])
    locs = label_frequencies.counts >= min_label_occurrences
    frequent_labels = label_frequencies.values[locs]
    label_edges = label_edges[label_edges.isin(labels, frequent_labels)]
    label_nodes = net.get_node(labels).loc(frequent_labels)
    label_map = dict(
        zip(label_nodes.index, np.arange(frequent_labels.shape[0]))
    )

    label_vec = np.zeros(
        (embeddings.shape[0], frequent_labels.shape[0]), np.bool_
    )
    label_vec[
        label_edges["Publication"],
        np.fromiter((label_map[l] for l in label_edges[labels]), dtype=int),
    ] = True

    return (embeddings, label_vec, label_nodes.index)
