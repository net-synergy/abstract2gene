__ALL__ = ["ArrayLike", "Batch", "PyTree", "LabelLike"]

from typing import Any, TypeAlias, Union

import numpy as np

ArrayLike: TypeAlias = Union[np.ndarray[Any, np.dtype[np.float32]]]
LabelLike: TypeAlias = Union[np.ndarray[Any, np.dtype[np.int_ | np.bool_]]]
Batch: TypeAlias = tuple[ArrayLike, ArrayLike, LabelLike]
PyTree: TypeAlias = dict[str, Any]
