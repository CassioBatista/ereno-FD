"""Util equivalent — ARFF loading and feature filtering."""
from __future__ import annotations

import numpy as np
import pandas as pd
from io import StringIO
from typing import Tuple
import python.config as config

# Normal class index (set after loading dataset — index of the first class value)
normal_class: int = 0


def load_arff(filepath: str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Load an ARFF file, returning (X, y, class_values).

    X: float64 array of shape (n_samples, n_features), features only
    y: int array of shape (n_samples,), class indices
    class_values: ordered list of class label strings
    """
    global normal_class
    attributes: list[tuple[str, str]] = []
    class_values: list[str] = []
    data_rows: list[str] = []
    in_data = False

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            low = stripped.lower()
            if low.startswith("@relation"):
                continue
            if low.startswith("@attribute"):
                # parse: @attribute <name> <type>
                parts = stripped.split(None, 2)  # [@attribute, name, type]
                attr_name = parts[1]
                attr_type = parts[2] if len(parts) > 2 else "numeric"
                if "{" in attr_type:
                    # Nominal: extract values
                    vals = attr_type.strip("{}").split(",")
                    class_values = [v.strip() for v in vals]
                    attributes.append((attr_name, "nominal"))
                else:
                    attributes.append((attr_name, "numeric"))
                continue
            if low.startswith("@data"):
                in_data = True
                continue
            if in_data:
                data_rows.append(stripped)

    if not data_rows:
        raise ValueError(f"No data rows found in {filepath}")

    n_attrs = len(attributes)
    n_features = n_attrs - 1  # last is class

    X_list: list[list[float]] = []
    y_list: list[int] = []

    for row in data_rows:
        vals = row.split(",")
        if len(vals) != n_attrs:
            continue
        try:
            x = [float(v.strip()) for v in vals[:-1]]
            cls_str = vals[-1].strip()
            if class_values:
                y_val = class_values.index(cls_str)
            else:
                y_val = int(float(cls_str))
            X_list.append(x)
            y_list.append(y_val)
        except (ValueError, IndexError):
            continue

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)

    # normal_class is the class index of the FIRST instance
    normal_class = int(y[0]) if len(y) > 0 else 0

    return X, y, class_values


def filter_features(X: np.ndarray, feature_indices: list[int]) -> np.ndarray:
    """Keep only the specified features (1-based indices → 0-based columns)."""
    if not feature_indices:
        return X
    cols = [i - 1 for i in sorted(feature_indices)]
    return X[:, cols]


def load_single_file(print_selection: bool = False) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Load dataset from config.DATASET."""
    X, y, class_values = load_arff(config.DATASET)
    return X, y, class_values


def load_and_filter_single_file(print_selection: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset and filter to config.FEATURE_SELECTION."""
    X, y, _ = load_single_file(print_selection)
    X = filter_features(X, config.FEATURE_SELECTION)
    if print_selection and config.FEATURE_SELECTION:
        print(f"{config.FEATURE_SELECTION} - {X.shape[1]} attributes in fact.")
    return X, y


def copy_and_filter(X: np.ndarray, y: np.ndarray, print_selection: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Filter a pre-loaded dataset by config.FEATURE_SELECTION."""
    Xf = filter_features(X, config.FEATURE_SELECTION)
    if print_selection and config.FEATURE_SELECTION:
        print(f"{config.FEATURE_SELECTION} - {Xf.shape[1]} attributes in fact.")
    return Xf, y


def get_result_average(results):
    """Average a list of Result objects (cross-validation folds)."""
    from python.result import Result
    return Result.average(results)
