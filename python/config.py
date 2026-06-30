"""GeneralParameters equivalent — global configuration."""
from __future__ import annotations
from typing import Optional

# Dataset path (set before running)
DATASET: str = ""

# GRASP / evaluation settings
FOLDS: int = 5
GRASP_SEED: int = 5
EVALUATION_SEED: int = 7
NUM_CLASSES: int = 2
NORMALIZE: bool = False
DEBUG_MODE: bool = False
CSV: bool = True
CROSS_VALIDATION: bool = True
SINGLE_FOLD_MODE: bool = False
PRINT_SELECTION: bool = False
PRINT_TRAINING_TIME: bool = False
PRINT_TESTING_TIME: bool = False

# GRASP algorithm names
GRASP_METHOD = ["GR-G-BF", "GR-G-VND", "GR-G-RVND", "F-G-VND", "F-G-RVND", "I-G-VND"]

# Runtime state (mutated during evaluation)
FEATURE_SELECTION: list[int] = []
SINGLE_CLASSIFIER_MODE = None  # ClassifierExtended set at runtime
CLASSIFIERS_FOREACH = None     # list of ClassifierExtended set at runtime

# Train/test split (used when CROSS_VALIDATION=False)
TRAIN = None
TEST = None
