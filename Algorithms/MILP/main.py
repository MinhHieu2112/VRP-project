"""
Algorithms/MILP/main.py
Entry-point cho MILP solver (PuLP/CBC) — sử dụng pipeline chuẩn hóa.
"""

import os
import sys
import json
import time

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PROJECT_ROOT)

from milp_solvers import solve_acvrp_milp
from Algorithms.Init_strategies.Init_strategies import init_solution
from Utils.Pipeline import load_data, build_result, save_result, visualize, KM_SCALE


def load_config() -> dict:
    path = os.path.join(CURRENT_DIR, 'config.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_upper_bound(matrix, demands_dict: dict,
                        capacity: int, max_vehicles: int) -> float:
    num_nodes = matrix.shape[0]
    solution  = init_solution(
        strategy     = "random",
        matrix       = matrix,
        num_nodes    = num_nodes,
        capacity     = capacity,
        demands      = demands_dict,
        max_vehicles = max_vehicles,
        validate     = False,
    )
    ub_units = sum(
        sum(matrix[r[i], r[i+1]] for i in range(len(r)-1))
        for r in solution
    )
    print(f"[MILP] Upper bound (NNH): {ub_units / KM_SCALE:.2f} km "
          f"| {len(solution)} xe")
    return float(ub_units)


def format_routes(routes_info: list) -> dict:
    routes = {}
    for idx, info in enumerate(routes_info):
        route = info['route']
        if route[-1] != 0:
            route.append(0)
        routes[idx] = route
    return routes


def main(limit_nodes):
    config = load_config()
    data   = load_data(config)

    matrix      = data['distance_matrix'][:limit_nodes, :limit_nodes]
    capacity    = data['vehicle_capacity']
    df_locs     = data['df_locations']
    demands_arr = data['demands'][:limit_nodes]
    num_nodes   = matrix.shape[0]

    demands_dict = {i: int(demands_arr[i]) for i in range(num_nodes)}

    cons      = config.get('global_constraints', {})
    milp_cfg  = config.get('solvers', {}).get('milp', {})
    max_v     = cons.get('max_vehicles', 200)
    timelimit = milp_cfg.get('max_runtime_seconds', 300)

    print(f"[MILP] {num_nodes} node | {max_v} xe | capacity={capacity} | "
          f"timelimit={timelimit}s")

    compute_upper_bound(matrix, demands_dict, capacity, max_v)

    print("--- Đang giải bằng MILP (PuLP/CBC) ---")
    start = time.time()
    status_str, obj_val_units, routes_info = solve_acvrp_milp(
        matrix, demands_dict,
        num_vehicles = max_v,
        capacity     = capacity,
        timelimit    = timelimit,
    )
    elapsed = time.time() - start

    print(f"\nTrạng thái solver: {status_str}")

    if obj_val_units is None:
        print("[MILP] Không tìm được nghiệm khả thi.")
        return

    routes = format_routes(routes_info)
    result = build_result("MILP", routes, obj_val_units, elapsed)

    save_result(result, config, "MILP")
    visualize(result, config, "MILP", df_locs)

    print(f"\n[MILP DONE] {result['total_distance_km']:.2f} km | "
          f"{result['num_vehicles']} xe | {elapsed:.2f}s")


if __name__ == "__main__":
    main(limit_nodes=100)