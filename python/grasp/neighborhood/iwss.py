"""IWSS — Incremental Wrapper-Based Subset Selection."""
from __future__ import annotations

from python.grasp.solution import GraspSolution
from python.grasp.neighborhood.structure import NeighborhoodStructure


class IWSS(NeighborhoodStructure):
    def __init__(self, grasp):
        self.grasp = grasp
        self._full_list: list[int] = []

    def _perform_add_movement(self, reference: GraspSolution, rcl_index: int) -> GraspSolution:
        reference.select_feature(rcl_index)
        return self.grasp.avaliar(reference)

    def run(self, seed: GraspSolution) -> GraspSolution:
        print("Running IWSS:")
        best_local = seed.new_clone(False)

        # Start from empty set
        while best_local.get_num_selected_features() > 0:
            best_local.deselect_feature(0)

        self._full_list = best_local.copy_rcl_features()

        for rcl_index in range(best_local.get_num_rcl_features() - 1, -1, -1):
            if self.grasp.current_time >= self.grasp.max_time:
                return best_local
            print(
                f"IWSS >>> adding {best_local.get_rcl_features()[rcl_index]}"
                f" > to set {best_local.get_feature_set()}"
                f"(Acc: {best_local.get_accuracy()}), (F1: {best_local.get_f1_score()})"
            )
            before = best_local.new_clone(False)
            add = self._perform_add_movement(before.new_clone(True), rcl_index)
            if add.is_better_than(best_local, self.grasp.criteria_metric):
                best_local = add.new_clone(False)

        self._restore_rcl(best_local)
        return best_local

    def _restore_rcl(self, best_local: GraspSolution) -> None:
        sol = set(best_local.get_selected_features())
        rcl = set(best_local.get_rcl_features())
        for f in self._full_list:
            if f not in sol and f not in rcl:
                best_local.add_feature_rcl(f)
