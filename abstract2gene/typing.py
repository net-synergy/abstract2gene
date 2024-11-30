from typing import TypeAlias, Union

import jax.numpy as jnp
from numpy.typing import NDArray

ArrayLike: TypeAlias = Union[jnp.ndarray, NDArray]
