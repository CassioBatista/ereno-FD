"""LocalSearches equivalent — VND and RVND local search strategies."""
from __future__ import annotations

import random
from python.grasp.solution import GraspSolution
from python.grasp.neighborhood.iwss import IWSS
from python.grasp.neighborhood.iwssr import IWSSr
from python.grasp.neighborhood.bit_flip import BitFlip


def busca_local(solution: GraspSolution, neighborhood, grasp) -> GraspSolution:
    best_local = neighborhood.run(solution)
    if best_local.is_better_than(solution, grasp.criteria_metric):
        return best_local
    return solution


def do_vnd(seed: GraspSolution, grasp) -> GraspSolution:
    structures = [IWSSr(grasp), IWSS(grasp)]
    print("Adding IWWSr...")
    print("Adding IWWS...")
    included_bf = False
    if grasp.num_bit_flip_features > 0 and not included_bf:
        structures.append(BitFlip(grasp))
        print("Adding BitFlip...")

    best_local = seed.new_clone(False)
    i = 0
    while i < len(structures):
        nova = busca_local(seed, structures[i], grasp)
        print(f"**** Selected bestLocal: {best_local.get_feature_set()}({best_local.get_f1_score()}), RCL:{best_local.get_rcl_features()}")
        print(f"Running structure: {i}")
        print(f"**** Selected nova: {nova.get_feature_set()}({nova.get_f1_score()}), RCL:{nova.get_rcl_features()}")

        if nova.is_better_than(best_local, grasp.criteria_metric):
            best_local = nova.new_clone(False)
            grasp.set_best_global_solution(nova.new_clone(False))

        grasp.num_bit_flip_features = best_local.get_num_selected_features()
        if grasp.num_bit_flip_features > 0 and not included_bf:
            included_bf = True
            structures.append(BitFlip(grasp))
            print("Adding BitFlip...")
        i += 1

    return best_local


def do_rvnd(seed: GraspSolution, grasp) -> GraspSolution:
    structures = [IWSSr(grasp), IWSS(grasp)]
    print("Adding IWWSr...")
    print("Adding IWWS...")
    if grasp.num_bit_flip_features > 0:
        structures.append(BitFlip(grasp))
        print("Adding BitFlip...")

    best = seed.new_clone(False)
    first_iteration = True

    while structures:
        try:
            idx = random.randint(0, len(structures) - 1)
            print(f"Running structure: {idx}")
            nova = busca_local(seed.new_clone(False), structures[idx], grasp)
            print(f"**** Selected melhor: {best.get_feature_set()}({best.get_f1_score()}), RCL:{best.get_rcl_features()}")
            print(f"**** Selected nova: {nova.get_feature_set()}({nova.get_f1_score()}), RCL:{nova.get_rcl_features()}")

            if nova.is_better_than(best, grasp.criteria_metric):
                if first_iteration:
                    first_iteration = False
                    structures.pop(idx)
                    print(f"REMOVE({idx}) - T: {len(structures)}")
                    grasp.num_bit_flip_features = nova.get_num_selected_features()
                else:
                    print("RESET")
                    best = nova.new_clone(False)
                    grasp.set_best_global_solution(best.new_clone(False))
                    structures = [IWSSr(grasp), IWSS(grasp)]
                    print("Adding IWWSr...")
                    print("Adding IWWS...")
                    if grasp.num_bit_flip_features > 0:
                        structures.append(BitFlip(grasp))
                        print("Adding BitFlip...")
                grasp.num_bit_flip_features = best.get_num_selected_features()
            else:
                structures.pop(idx)
                print(f"REMOVE({idx}) - T: {len(structures)}")
                grasp.num_bit_flip_features = best.get_num_selected_features()
        except IndexError as e:
            print(f"Sorteou numero invalido, cancelando a rodada: {e}")

    return best
