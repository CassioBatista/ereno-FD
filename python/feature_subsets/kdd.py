"""KddFeatures — KDD dataset feature subset definitions."""
from python.feature_subsets.base import FeatureSubsets

KDD_FULL = list(range(1, 42))
KDD_RCL_GR = list(range(1, 42))

RCL_KDD_IWSSR_RandomTree   = [1, 5, 23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
RCL_KDD_IWSSR_J48          = [1, 5, 23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
RCL_KDD_IWSSR_RepTree      = [1, 5, 23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
RCL_KDD_IWSSR_NaiveBayes   = [1, 5, 23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
RCL_KDD_IWSSR_RandomForest = [1, 5, 23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
RCL_KDD_IWSSR = [
    RCL_KDD_IWSSR_RandomTree,
    RCL_KDD_IWSSR_J48,
    RCL_KDD_IWSSR_RepTree,
    RCL_KDD_IWSSR_NaiveBayes,
    RCL_KDD_IWSSR_RandomForest,
]


class KddFeatures(FeatureSubsets):
    def __init__(self):
        super().__init__(KDD_FULL, KDD_RCL_GR, RCL_KDD_IWSSR)
