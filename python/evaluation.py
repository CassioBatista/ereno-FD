"""GenericEvaluation equivalent — train and test a single classifier."""
from __future__ import annotations

import time
import numpy as np
from sklearn.base import clone

import python.config as config
from python.result import Result
import python.util as util


def _run_classifier(classifier_ext, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray) -> Result:
    """Train classifier on train set and evaluate on test set."""
    clf = clone(classifier_ext.get_classifier())

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t1 = time.perf_counter()

    VP = VN = FP = FN = 0
    n_classes = config.NUM_CLASSES
    confusion = [[0] * n_classes for _ in range(n_classes)]

    t_pred_start = time.perf_counter()
    y_pred = clf.predict(X_test)
    t_pred_end = time.perf_counter()

    nc = util.normal_class
    for expected, predicted in zip(y_test, y_pred):
        if predicted == expected:
            if predicted == nc:
                VN += 1
            else:
                VP += 1
        else:
            if predicted == nc:
                FN += 1
            else:
                FP += 1
        if expected < n_classes and predicted < n_classes:
            confusion[int(expected)][int(predicted)] += 1

    n_test = len(y_test)
    avg_time_ns = (t_pred_end - t_pred_start) * 1e9 / n_test if n_test else 0.0

    if config.PRINT_TRAINING_TIME:
        print(f"Tempo de treinamento = {(t1 - t0) * 1e9:.0f} ns")

    return Result(classifier_ext.get_classifier_name(), VP, FN, VN, FP, avg_time_ns, confusion)


def run_single_classifier(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Result:
    clf = config.SINGLE_CLASSIFIER_MODE
    r = _run_classifier(clf, X_train, y_train, X_test, y_test)
    if config.CSV:
        print(
            f"{r.cx};{str(r.accuracy).replace(',', '.')};{str(r.precision).replace(',', '.')};"
            f"{str(r.recall).replace(',', '.')};{str(r.f1score).replace(',', '.')};"
            f"{r.VP};{r.VN};{r.FP};{r.FN};{r.avg_time}"
        )
    return r


def run_multiple_classifier(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> list[Result]:
    results = []
    for clf_ext in config.CLASSIFIERS_FOREACH:
        r = _run_classifier(clf_ext, X_train, y_train, X_test, y_test)
        results.append(r)
        if config.CSV:
            print(
                f"{clf_ext.get_classifier_name()};"
                f"{str(r.accuracy).replace(',', '.')};{str(r.precision).replace(',', '.')};"
                f"{str(r.recall).replace(',', '.')};{str(r.f1score).replace(',', '.')};"
                f"{r.VP};{r.VN};{r.FP};{r.FN};{r.avg_time}"
            )
    return results
