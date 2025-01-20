__all__ = ["Fetures", "Labels", "Batch", "PyTree", "Names"]

from typing import Any, Sequence

import jax

Samples = jax.Array
Labels = jax.Array
Batch = tuple[Samples, Labels]
PyTree = dict[str, Any]
Names = Sequence[str]
