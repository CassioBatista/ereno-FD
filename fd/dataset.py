"""Partition an ARFF dataset into N client splits (IID)."""

import numpy as np
import python.util as util


def load_and_partition(
    filepath: str,
    feature_indices: list[int],
    num_clients: int,
    seed: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Load ARFF, apply feature selection, return per-client (X, y) partitions.

    Returns (partitions, X_all, y_all).  X_all / y_all are the full arrays
    (useful for centralized baseline evaluation on the same data).
    """
    X_all, y_all, _ = util.load_arff(filepath)

    cols = [i - 1 for i in sorted(feature_indices)]
    Xf = X_all[:, cols]

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(Xf))
    splits = np.array_split(indices, num_clients)

    partitions = [(Xf[idx], y_all[idx]) for idx in splits]
    return partitions, Xf, y_all
