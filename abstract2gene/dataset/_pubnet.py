"""Create DataSets from a pubnet."""

__all__ = ["net2dataset"]

import numpy as np
from pubnet import PubNet

from ._dataset import DataSet


def net2dataset(
    net: PubNet,
    features: str = "Abstract_Embedding",
    labels: str = "Gene",
    label_name: str = "GeneSymbol",
    feature_name: str = "PMID",
    remove_baseline: bool = False,
    min_occurrences: int = 50,
    **kwds,
) -> DataSet:
    """Return the network as a matrix of features and a matrix of labels.

    Features are normalized by publication so each publication's features has
    an l2 norm of 1.

    Features are samples (publications) x features and labels are samples x
    labels (binary vector).

    Label IDs is an array of the label nodes index that are in used in the
    labels. Labels are limited to genes with at least `min_occurrences`.

    If `remove_baseline` subtract the average of each feature.

    Any other keyword arguments will be based on to the DataSet constructor.
    """
    feature_names = net.get_node("Publication").feature_vector(feature_name)
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
    locs = label_frequencies.counts >= min_occurrences
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
        np.fromiter((label_map[y] for y in label_edges[labels]), dtype=int),
    ] = True

    label_names = label_nodes.feature_vector(label_name)

    return DataSet(
        embeddings,
        label_vec,
        feature_names,
        label_names,
        **kwds,
    )
