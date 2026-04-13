"""
Algorithms/SA/main.py
Entry-point cho Simulated Annealing solver — sử dụng pipeline chuẩn hóa.
"""

import os
import sys
import json
import time

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PROJECT_ROOT)

from solver_sa import SimulatedAnnealingSolver
from Utils.Pipeline import load_data, build_result, save_result, visualize


def load_config() -> dict:
    """Đọc config.json của SA."""
    path = os.path.join(CURRENT_DIR, 'config.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_sa():
    """Chạy toàn bộ pipeline SA: load → solve → save → visualize."""
    print("\n===== RUN SIMULATED ANNEALING =====")
    config = load_config()
    data   = load_data(config)

    solver = SimulatedAnnealingSolver(data, config)

    start = time.time()
    routes, total_cost_units = solver.solve()   # đơn vị nội bộ (matrix_int)
    elapsed = time.time() - start

    result = build_result("SA", routes, total_cost_units, elapsed)

    save_result(result, config, "SA")
    visualize(result, config, "SA", data['df_locations'])

    print(f"\n[SA DONE] {result['total_distance_km']:.2f} km | "
          f"{result['num_vehicles']} xe | {elapsed:.2f}s")
    return result


if __name__ == "__main__":
    run_sa()