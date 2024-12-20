"""Create DataSets from a pubnet."""

__all__ = ["net2dataset"]

import numpy as np
import scipy as sp
from pubnet import PubNet

from ._dataset import DataSet


def net2dataset(
    net: PubNet,
    features: str = "Abstract_Embedding",
    labels: str = "Gene",
    label_name: str = "GeneSymbol",
    feature_name: str = "PMID",
    remove_baseline: bool = False,
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
    net.repack()

    sample_names = net.get_node("Publication").feature_vector(feature_name)
    embeddings_edge = net.get_edge(features, "Publication")
    n_features = np.sum(embeddings_edge["Publication"] == 0)
    embeddings = embeddings_edge.feature_vector("embedding").reshape(
        (-1, n_features)
    )

    if remove_baseline:
        baseline = embeddings.mean(axis=0, keepdims=True)
        embeddings = embeddings - baseline

    label_edges = net.get_edge("Publication", labels)
    label_data = np.ones((len(label_edges),), dtype=np.bool)
    label_names = net.get_node(labels).feature_vector(label_name)

    _labels = sp.sparse.coo_array(
        (label_data, (label_edges["Publication"], label_edges[labels])),
        (embeddings.shape[0], label_names.shape[0]),
    ).tocsc()

    return DataSet(
        embeddings,
        _labels,
        sample_names,
        label_names,
        **kwds,
    )
