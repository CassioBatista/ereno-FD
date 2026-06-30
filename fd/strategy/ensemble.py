"""EnsembleStrategy — aggregate client models into a majority-vote ensemble."""

from logging import WARNING
from typing import Optional, Union

import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fd.model import deserialize_model, serialize_model


class EnsembleStrategy(FedAvg):
    """Each client sends its fully trained sklearn model.

    The server wraps all received models in a lightweight majority-vote
    ensemble and broadcasts it back (serialised) so future evaluation
    rounds can run on the aggregated model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_models: list = []

    # ------------------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        if not results:
            return None, {}

        self.global_models = []
        for _, fit_res in results:
            model = deserialize_model(parameters_to_ndarrays(fit_res.parameters))
            self.global_models.append(model)

        # Serialise a dummy placeholder (the real ensemble lives in self.global_models).
        # We send back the first model so Flower is satisfied — evaluation uses
        # majority_vote_predict() directly.
        aggregated_params = ndarrays_to_parameters(serialize_model(self.global_models[0]))
        metrics: dict[str, Scalar] = {"num_models": len(self.global_models)}
        return aggregated_params, metrics

    # ------------------------------------------------------------------
    def majority_vote_predict(self, X: np.ndarray) -> np.ndarray:
        """Run all client models and return majority-vote predictions."""
        if not self.global_models:
            raise RuntimeError("No models aggregated yet — call aggregate_fit first.")
        votes = np.stack([m.predict(X) for m in self.global_models], axis=0)
        # axis=0 → majority across models per sample
        result = np.apply_along_axis(
            lambda col: np.bincount(col.astype(int)).argmax(), 0, votes
        )
        return result
