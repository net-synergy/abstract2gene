__all__ = ["lol_to_csc"]

import numpy as np
import scipy as sp


def lol_to_csc(lists: list[list[int]]) -> sp.sparse.sparray:
    """Convert a list of lists to a binary csc sparse array."""
    nrows = len(lists)
    ncols = max((col for ls in lists for col in ls)) + 1
    numel = sum((len(ls) for ls in lists))
    coords = np.zeros((2, numel))

    count = 0
    for i, ls in enumerate(lists):
        for el in ls:
            coords[:, count] = [i, el]
            count += 1

    return sp.sparse.coo_array(
        (np.ones((numel,), dtype=np.bool), coords), shape=(nrows, ncols)
    ).tocsc()
