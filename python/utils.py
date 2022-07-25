import numpy as np


def to_floats(xs: np.ndarray, max_val: int = 3) -> np.ndarray:
    # x from 0 to max_val
    fracs = xs / max_val
    return fracs


def to_ints(xs: np.ndarray, max_val: int = 3) -> np.ndarray:
    # x from 0 to max_val
    ints = np.rint((xs * max_val)).astype(int)
    return ints
