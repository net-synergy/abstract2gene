__all__ = ["Fetures", "Batch", "PyTree", "Labels"]

from typing import Any

import jax
import numpy as np

Features = jax.Array
Labels = jax.Array
Batch = tuple[Features, Features, Labels]
PyTree = dict[str, Any]
Names = np.ndarray[Any, np.dtype[np.str_ | np.object_]]
