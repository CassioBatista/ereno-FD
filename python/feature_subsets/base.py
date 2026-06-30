"""FeatureSubsets base class — RCL definitions for each dataset."""
from __future__ import annotations


class FeatureSubsets:
    RCL_FULL: list[int] = []
    RCL_GR: list[int] = []
    RCL_I: list[list[int]] = []

    WSN_FULL = list(range(1, 19))
    CICIDS_FULL = list(range(1, 79))
    KDD_FULL = list(range(1, 42))
    SWAT = list(range(1, 52))

    def __init__(self, rcl_full: list[int], rcl_gr: list[int], rcl_i: list[list[int]]):
        FeatureSubsets.RCL_FULL = rcl_full
        FeatureSubsets.RCL_GR = rcl_gr
        FeatureSubsets.RCL_I = rcl_i
