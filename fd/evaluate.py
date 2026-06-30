"""Unified evaluation: compute VP/VN/FP/FN from predictions and return Result."""

import numpy as np
import python.util as util
from python.result import Result


def evaluate_predictions(
    classifier_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Result:
    """Convert raw predictions to a Result object using the same VP/FN/VN/FP
    logic as python/evaluation.py so results are directly comparable."""
    nc = util.normal_class
    VP = VN = FP = FN = 0

    for expected, predicted in zip(y_true, y_pred):
        correct   = bool(predicted == expected)
        is_normal = bool(predicted == nc)
        match (correct, is_normal):
            case (True,  True):  VN += 1
            case (True,  False): VP += 1
            case (False, True):  FN += 1
            case (False, False): FP += 1

    return Result(classifier_name, VP, FN, VN, FP)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
) -> Result:
    y_pred = model.predict(X_test)
    return evaluate_predictions(name, y_test, y_pred)
