"""GraspVND — GRASP with Variable Neighborhood Descent."""
from __future__ import annotations

import time
from python.grasp.base import Grasp
from python.grasp.solution import GraspSolution
from python.grasp.local_searches import do_vnd


class GraspVND(Grasp):

    def run(self, rcl: list[int], method_name: str, dataset: str) -> GraspSolution:
        self.begin_time = time.time() * 1000
        print(f"######### ITERATION ({self.iteration_number}) #############")

        full_rcl = self.build_custom_rcl(rcl)
        initial = self.construct_solution(full_rcl)
        initial = self.avaliar(initial)
        self.set_best_global_solution(initial.new_clone(False))

        initial = do_vnd(initial, self)
        self.iteration_number += 1
        if initial.is_better_than(self.get_best_global_solution(), self.criteria_metric):
            self.set_best_global_solution(initial.new_clone(False))

        while (
            self.iteration_number < self.max_iterations
            and self.no_improvements < self.max_no_improvement
            and self.current_time < self.max_time
        ):
            if self.number_evaluation >= self.max_number_evaluation:
                return self.get_best_global_solution()

            self.iteration_number += 1
            print(f"######### ITERATION ({self.iteration_number}) #############")

            reconstructed = self.reconstruct_solution(initial)
            self.avaliar(reconstructed)

            reconstructed = do_vnd(reconstructed, self)
            if reconstructed.is_better_than(self.get_best_global_solution(), self.criteria_metric):
                self.set_best_global_solution(reconstructed.new_clone(False))
                print(
                    f"GLOBAL IMPROVEMENT: {self.get_best_global_solution().get_feature_set()}"
                    f" = {self.get_best_global_solution().evaluation.f1score if self.get_best_global_solution().evaluation else 'N/A'}"
                )
                self.no_improvements = 0
            else:
                print(f"Sem melhoras: {self.no_improvements} atualizado para ", end="")
                self.no_improvements += 1
                print(f"Sem melhoras: {self.no_improvements}")

            self.current_time = time.time() * 1000 - self.begin_time
            best = self.get_best_global_solution()
            print(
                f"######### Fim ITERAÇÂO ({self.iteration_number}"
                f" / Current Time:{self.current_time / 1000 / 60:.1f}min)"
                f" - F1Score:{best.evaluation.f1score if best.evaluation else 'N/A'}"
                f" - Conjunto = {best.get_feature_set()}"
            )

        return self.get_best_global_solution()
