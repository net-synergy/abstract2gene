"""Analyze gene predictions."""

__all__ = ["Gene", "top_predictions"]

from typing import TypeAlias

import numpy as np

Gene: TypeAlias = dict[str, list[str]]


def top_predictions(
    predictions: list[float], genes: Gene, k: int, p: float
) -> dict[str, list[str]]:
    """Find the top highest predicted genes.

    Returns all genes with predictions greater than p or a minimum of k genes
    if less than k genes have a prediction greater than p.
    """
    preds = np.asarray(predictions)
    indices = np.argsort(preds)[::-1]

    if sum(preds > p) < k:
        indices = indices[:k]
    else:
        indices = indices[preds[indices] > p]

    return {
        "symbol": [genes["symbol"][idx] for idx in indices],
        "entrez_id": [genes["entrez_id"][idx] for idx in indices],
        "prediction": [f"{predictions[idx]:.2f}" for idx in indices],
    }
