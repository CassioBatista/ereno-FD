"""CicidsFeatures — CICIDS dataset feature subset definitions."""
from python.feature_subsets.base import FeatureSubsets

CICIDS_FULL = list(range(1, 79))
CICIDS_RCL_GR = list(range(1, 79))

GRASP_VND_CICIDS_IWSSR_RandomTree   = [68, 78, 67, 56, 49, 43, 1]
GRASP_VND_CICIDS_IWSSR_J48          = [49, 1, 6, 10, 16, 25, 39, 67, 68, 75, 47]
GRASP_VND_CICIDS_IWSSR_NaiveBayes   = [28, 31, 26, 24, 23, 8, 1]
GRASP_VND_CICIDS_IWSSR_RandomForest = [25, 78, 68, 67, 56, 52, 11, 7, 6, 1]
GRASP_VND_CICIDS_IWSSR_RepTree      = [1, 10, 25, 49, 66, 67, 68, 78, 71]
GRASP_VND_CICIDS_IWSSR = [
    GRASP_VND_CICIDS_IWSSR_RandomTree,
    GRASP_VND_CICIDS_IWSSR_J48,
    GRASP_VND_CICIDS_IWSSR_RepTree,
    GRASP_VND_CICIDS_IWSSR_NaiveBayes,
    GRASP_VND_CICIDS_IWSSR_RandomForest,
]

RCL_GR = CICIDS_RCL_GR


class CicidsFeatures(FeatureSubsets):
    def __init__(self):
        super().__init__(CICIDS_FULL, RCL_GR, GRASP_VND_CICIDS_IWSSR)
