__all__ = ["Fetures", "Labels", "Batch", "PyTree", "Names"]

from typing import Any, Sequence

import jax
import numpy as np

Features = jax.Array
Labels = jax.Array
Batch = tuple[Features, Labels]
PyTree = dict[str, Any]
Names = Sequence[str]
