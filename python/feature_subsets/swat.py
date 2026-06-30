"""SWATFeatures — SWAT dataset feature subset definitions."""
from python.feature_subsets.base import FeatureSubsets

SWAT_FULL = list(range(1, 52))


class SWATFeatures(FeatureSubsets):
    def __init__(self):
        super().__init__(SWAT_FULL, SWAT_FULL, [SWAT_FULL])
