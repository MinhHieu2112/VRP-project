"""
Algorithms/Tabu/main_tabu.py
Entry-point cho Granular Tabu Search solver.
Cải tiến dựa trên Toth & Vigo (2003) và Gendreau et al. (1994).
"""

import os
import sys
import json
import time
import numpy as np

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PROJECT_ROOT)

from Algorithms.Init_strategies.Init_strategies import init_solution
from Algorithms.Tabu.tabu_solver import GranularTabuSearch
from Utils.Pipeline import load_data, build_result, save_result, visualize


def load_config() -> dict:
    path = os.path.join(CURRENT_DIR, 'config_tabu.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def verify_coverage(initial_state: list, num_nodes: int):
    served  = {n for r in initial_state for n in r if n != 0}
    missing = set(range(1, num_nodes)) - served
    if missing:
        print(f"[WARN] {len(missing)} KH chưa được phục vụ sau init!")
    else:
        print(f"[OK] Init: {len(initial_state)} xe, "
              f"tất cả {num_nodes-1} KH được phục vụ")


def run_tabu():
    print("\n===== RUN GRANULAR TABU SEARCH (Toth & Vigo, 2003) =====")
    config = load_config()
    data   = load_data(config)

    matrix    = data['distance_matrix']
    capacity  = data['vehicle_capacity']
    df_locs   = data['df_locations']
    demands_arr = data['demands']
    num_nodes = matrix.shape[0]

    demands_dict = {i: int(demands_arr[i]) for i in range(num_nodes)}
    tabu_p  = config['tabu_parameters']
    cons    = config['constraints']

    # Khởi tạo bằng Clarke-Wright (tốt hơn random cho Tabu)
    initial_state = init_solution(
        strategy      = "greedy",
        matrix        = matrix,
        num_nodes     = num_nodes,
        capacity      = capacity,
        demands       = demands_dict,
        max_vehicles  = cons['max_vehicles'],
        validate      = True,
    )
    verify_coverage(initial_state, num_nodes)

    solver = GranularTabuSearch(
        distance_matrix = matrix,
        demands         = demands_dict,
        capacity        = capacity,
        max_v           = cons['max_vehicles'],
        tabu_size       = tabu_p.get('tabu_size',        20),
        max_iter        = tabu_p.get('max_iterations', 5000),
        max_no_improve  = tabu_p.get('max_no_improve',  300),
        granular_beta   = tabu_p.get('granular_beta',   1.5),
        granular_k      = tabu_p.get('granular_k',       20),
        penalty_lambda  = tabu_p.get('penalty_lambda',   1.0),
        penalty_h       = tabu_p.get('penalty_h',        10),
    )

    start = time.time()
    best_state, best_cost_units = solver.solve(initial_state)
    elapsed = time.time() - start

    result = build_result("Granular Tabu Search", best_state, best_cost_units, elapsed)
    save_result(result, config, "Tabu")
    visualize(result, config, "Tabu", df_locs)

    print(f"\n[TABU DONE] {result['total_distance_km']:.2f} km | "
          f"{result['num_vehicles']} xe | {elapsed:.2f}s")
    return result


if __name__ == "__main__":
    run_tabu()