"""Models for comparing abstract features.

Various models for comparing abstract embeddings. This generally means
comparing an individual publication's abstract embeddings to template
embeddings for different labels (in the case of this package, genes).

Templates are the average of many examples of abstracts tagged with a label.
"""

__all__ = [
    "RawSimilarity",
    "MultiLayer",
    "MLPExtras",
    "Attention",
    "Trainer",
    "Model",
]

from ._models import Attention, MLPExtras, Model, MultiLayer, RawSimilarity
from ._trainer import Trainer
