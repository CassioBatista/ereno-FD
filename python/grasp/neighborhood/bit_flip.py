"""BitFlip neighborhood structure — randomly swap one selected feature with one RCL feature."""
from __future__ import annotations

import random
from python.grasp.solution import GraspSolution
from python.grasp.neighborhood.structure import NeighborhoodStructure


class BitFlip(NeighborhoodStructure):
    def __init__(self, grasp):
        self.grasp = grasp
        self._rem_ls_iterations = 50
        self._rem_ls_no_improvements = 20

    def _perform_single_movement(self, reference: GraspSolution) -> GraspSolution:
        neighbor = reference.new_clone(True)
        n_sol = neighbor.get_num_selected_features()
        n_rcl = neighbor.get_num_rcl_features()
        if n_sol == 0 or n_rcl == 0:
            raise ValueError(f"IncompleteFeatureSelection: sol={n_sol}, rcl={n_rcl}")
        rem = 0 if n_sol == 1 else random.randint(0, n_sol - 2)
        add = 0 if n_rcl == 1 else random.randint(0, n_rcl - 2)
        neighbor.replace_feature(rem, add)
        neighbor = self.grasp.avaliar(neighbor)
        return neighbor

    def run(self, reference: GraspSolution) -> GraspSolution:
        print("Running BitFlip:")
        best_local = reference.new_clone(False)
        iterations = self._rem_ls_iterations
        no_imp = self._rem_ls_no_improvements

        while iterations > 0 and no_imp > 0:
            iterations -= 1
            no_imp -= 1
            try:
                neighbor = self._perform_single_movement(reference)
                if neighbor.is_better_than(best_local, self.grasp.criteria_metric):
                    best_local = neighbor.new_clone(False)
                    no_imp = 10
            except ValueError as e:
                print(f"Cancelando run: {e}")
                return best_local

        return best_local
