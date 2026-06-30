"""ERENO-FD — Federated IDS entry point.

Usage:
    python main_fd.py <strategy> <classifier_idx> <dataset_name> [<num_clients>]

    strategy       : ensemble | federated_nb
    classifier_idx : 1=RandomTree  2=J48  3=REPTree  4=NaiveBayes  5=RandomForest
    dataset_name   : ARFF file without .csv  (e.g. all_in_one_wsn)
    num_clients    : number of federated clients  (default: 3)

Examples:
    python main_fd.py ensemble      2 all_in_one_wsn 3
    python main_fd.py federated_nb  4 all_in_one_wsn 5
"""

import sys
import os
import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import python.config as config
import python.util   as util
from python.classifiers import all_classifiers

from fd.dataset  import load_and_partition
from fd.client   import ErenoClient
from fd.evaluate import evaluate_model, evaluate_predictions
from fd.strategy.ensemble     import EnsembleStrategy
from fd.strategy.federated_nb import FederatedNBStrategy
from fd.model import nb_to_params, serialize_model


# ── main ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    strategy_name = sys.argv[1].lower()
    clf_idx       = int(sys.argv[2]) - 1   # 0-based index into all_classifiers
    dataset_name  = sys.argv[3]
    num_clients   = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    dataset_path  = f"{dataset_name}.csv"
    if not os.path.exists(dataset_path):
        sys.exit(f"Dataset não encontrado: {dataset_path}")

    clf_ext  = all_classifiers[clf_idx]
    base_clf = clf_ext.get_classifier()

    # ── load & partition ──────────────────────────────────────────────────
    X_all, y_all, _ = util.load_arff(dataset_path)

    partitions, _, _ = load_and_partition(
        dataset_path, list(range(1, X_all.shape[1] + 1)),
        num_clients, seed=config.EVALUATION_SEED,
    )

    # Global 80/20 train/test split (same data seen by both runs)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=config.EVALUATION_SEED, stratify=y_all
    )

    # Per-client 80/20 splits
    client_splits: list[tuple] = []
    for X_c, y_c in partitions:
        X_ctr, X_cte, y_ctr, y_cte = train_test_split(
            X_c, y_c, test_size=0.2, random_state=config.EVALUATION_SEED
        )
        client_splits.append((X_ctr, y_ctr, X_cte, y_cte))

    # ── validate federated_nb ─────────────────────────────────────────────
    if strategy_name == "federated_nb":
        from sklearn.naive_bayes import GaussianNB
        if not isinstance(base_clf, GaussianNB):
            print(
                f"[AVISO] federated_nb requer NaiveBayes (classifier 4). "
                f"Classificador {clf_idx + 1} ({clf_ext.get_classifier_name()}) "
                f"substituído por GaussianNB."
            )
            base_clf = GaussianNB()

    # ── build strategy ────────────────────────────────────────────────────
    fit_config = {"mode": strategy_name}

    match strategy_name:
        case "federated_nb":
            from sklearn.naive_bayes import GaussianNB
            _dummy = GaussianNB()
            _dummy.fit(X_tr[:20], y_tr[:20])
            strategy = FederatedNBStrategy(
                min_fit_clients=num_clients,
                min_evaluate_clients=num_clients,
                min_available_clients=num_clients,
                on_fit_config_fn=lambda _: fit_config,
                initial_parameters=ndarrays_to_parameters(nb_to_params(_dummy)),
            )
        case _:  # ensemble
            _dummy_clf = clone(base_clf)
            _dummy_clf.fit(X_tr[:20], y_tr[:20])
            strategy = EnsembleStrategy(
                min_fit_clients=num_clients,
                min_evaluate_clients=num_clients,
                min_available_clients=num_clients,
                on_fit_config_fn=lambda _: fit_config,
                initial_parameters=ndarrays_to_parameters(serialize_model(_dummy_clf)),
            )

    def client_fn(cid: str) -> fl.client.Client:
        i = int(cid)
        X_ctr, y_ctr, X_cte, y_cte = client_splits[i]
        return ErenoClient(i, X_ctr, y_ctr, X_cte, y_cte, base_clf).to_client()

    # ── simulation ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ERENO-FD")
    print(f"  estratégia  : {strategy_name}")
    print(f"  clientes    : {num_clients}")
    print(f"  classificador: {clf_ext.get_classifier_name()}")
    print(f"  dataset     : {dataset_name}")
    print(f"{'='*60}\n")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )

    # ── evaluate global federated model ───────────────────────────────────
    print(f"\n{'='*60}")
    print("  Avaliação no conjunto de teste global (80/20 split)")
    print(f"{'='*60}")

    match strategy_name:
        case "federated_nb":
            global_model = strategy.get_global_model()
            if global_model is None:
                sys.exit("Agregação falhou — nenhum modelo global disponível.")
            r_fed = evaluate_model(global_model, X_te, y_te, "FederatedNB")
            fed_label = "FEDERADO (NaiveBayes exato)"
        case _:
            y_pred = strategy.majority_vote_predict(X_te)
            r_fed  = evaluate_predictions("Ensemble", y_te, y_pred)
            fed_label = f"FEDERADO (Ensemble c/ {num_clients} clientes)"

    _print_result(fed_label, r_fed)

    # ── centralized baseline ──────────────────────────────────────────────
    clf_central = clone(base_clf)
    clf_central.fit(X_tr, y_tr)
    r_central = evaluate_model(clf_central, X_te, y_te, clf_ext.get_classifier_name())
    _print_result(f"CENTRALIZADO — {clf_ext.get_classifier_name()}", r_central)

    # ── delta ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Diferença  (federado − centralizado)")
    print(f"{'='*60}")
    for attr, label in [("f1score", "F1-score"), ("accuracy", "Accuracy"),
                        ("recall", "Recall   "), ("precision", "Precision")]:
        delta = getattr(r_fed, attr) - getattr(r_central, attr)
        print(f"  {label}: {delta:+.4f} pp")


# ── helpers ────────────────────────────────────────────────────────────────

def _print_result(label: str, r) -> None:
    print(f"\n  [{label}]")
    print(f"    Accuracy : {r.accuracy:.4f}%")
    print(f"    Precision: {r.precision:.4f}%")
    print(f"    Recall   : {r.recall:.4f}%")
    print(f"    F1-score : {r.f1score:.4f}%")
    print(f"    VP={r.VP}  VN={r.VN}  FP={r.FP}  FN={r.FN}")


if __name__ == "__main__":
    main()
