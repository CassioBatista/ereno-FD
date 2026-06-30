"""FederatedNBStrategy — mathematically exact NaiveBayes federation.

Clients send (theta_, var_, class_count_).  The server combines them using
the parallel mean / pooled variance formulas, producing a GaussianNB that
is numerically identical to one trained on all data centrally.
"""

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

from fd.model import nb_to_params, params_to_nb
from sklearn.naive_bayes import GaussianNB


class FederatedNBStrategy(FedAvg):
    """Aggregate NaiveBayes statistics instead of model weights."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_model: Optional[GaussianNB] = None

    # ------------------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        if not results:
            return None, {}

        # Collect (theta, var, class_count) triples from all clients.
        stats: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for _, fit_res in results:
            theta, var, cc = parameters_to_ndarrays(fit_res.parameters)
            stats.append((theta, var, cc))

        # --- Parallel mean --------------------------------------------------
        # theta_[k, d] = weighted mean of per-class feature means
        total_cc = sum(s[2] for s in stats)            # shape (K,)
        theta_global = (
            sum(s[2][:, None] * s[0] for s in stats) / total_cc[:, None]
        )                                              # shape (K, D)

        # --- Pooled variance (Chan's parallel formula) ----------------------
        # var_global[k, d] = Σ n_i * (var_i + (mu_i - mu_global)^2) / Σ n_i
        var_global = (
            sum(
                s[2][:, None] * (s[1] + (s[0] - theta_global) ** 2)
                for s in stats
            )
            / total_cc[:, None]
        )                                              # shape (K, D)

        self.global_model = params_to_nb(theta_global, var_global, total_cc)

        aggregated_params = ndarrays_to_parameters(
            nb_to_params(self.global_model)
        )
        return aggregated_params, {"clients": len(stats)}

    # ------------------------------------------------------------------
    def get_global_model(self) -> Optional[GaussianNB]:
        return self.global_model
