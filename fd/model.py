"""Model serialization helpers and NaiveBayes statistics extraction."""

import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from flwr.common import NDArrays


def serialize_model(model) -> NDArrays:
    """Pickle a sklearn model into a single-element NDArrays."""
    return [np.frombuffer(pickle.dumps(model), dtype=np.uint8)]


def deserialize_model(params: NDArrays):
    """Restore a sklearn model from an NDArrays produced by serialize_model."""
    return pickle.loads(params[0].tobytes())


def nb_to_params(model: GaussianNB) -> NDArrays:
    """Extract GaussianNB sufficient statistics for exact federation.

    Returns [theta_, var_, class_count_] as numpy arrays.
    """
    return [model.theta_.copy(), model.var_.copy(), model.class_count_.copy()]


def params_to_nb(theta: np.ndarray, var: np.ndarray, class_count: np.ndarray) -> GaussianNB:
    """Reconstruct a GaussianNB from aggregated statistics."""
    nb = GaussianNB()
    nb.theta_ = theta
    nb.var_ = var
    nb.class_count_ = class_count
    nb.classes_ = np.arange(len(class_count))
    nb.class_prior_ = class_count / class_count.sum()
    return nb
