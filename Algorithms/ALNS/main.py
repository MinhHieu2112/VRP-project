"""
Algorithms/ALNS/main.py
Entry-point cho ALNS solver — sử dụng pipeline chuẩn hóa.
"""

import os
import sys
import json
import time
import threading
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Algorithms.Init_strategies.Init_strategies import init_solution, _build_demands
from alns.stop import MaxIterations
from src.state import CvrpState
from src.solver import configure_alns
from Utils.Pipeline import load_data, build_result, save_result, visualize, KM_SCALE


# ── Stopping criterion ───────────────────────────────────────────────────────

class NoImprovementStop:
    """Dừng khi không cải thiện sau max_no_improve vòng liên tiếp."""

    def __init__(self, max_no_improve: int):
        self._limit  = max_no_improve
        self._count  = 0
        self._best   = float('inf')

    def __call__(self, rng, best, curr) -> bool:
        obj = best.objective()
        if obj < self._best - 1e-6:
            self._best  = obj
            self._count = 0
        else:
            self._count += 1
        return self._count >= self._limit


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str = 'config.json') -> dict:
    """Đọc config.json của ALNS."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_initial_state(matrix: np.ndarray, capacity: int,
                         demands: np.ndarray, config: dict) -> CvrpState:
    """Tạo nghiệm ban đầu bằng capacity_random và bọc vào CvrpState."""
    num_nodes    = matrix.shape[0]
    demands_dict = {i: int(demands[i]) for i in range(num_nodes)}

    routes = init_solution(
        strategy     = "random",
        matrix       = matrix,
        num_nodes    = num_nodes,
        capacity     = capacity,
        demands      = demands_dict,
        default_demand = 1.0,
        max_vehicles = config.get('constraints', {}).get('max_vehicles', 200),
        validate     = True,
    )
    return CvrpState(routes, [], matrix, capacity, demands_dict, config)


def make_progress_logger(stop_flag, shared, start_time):
    """Thread in tiến độ mỗi 10 giây."""
    def _run():
        while not stop_flag.is_set():
            stop_flag.wait(10)
            if stop_flag.is_set():
                break
            elapsed = time.time() - start_time
            sys.stdout.write(
                f"\r  [{elapsed:5.0f}s] Best: {shared[0]:.2f} km | "
                f"Unassigned: {shared[1]} | Cải thiện: {shared[2]} lần   "
            )
            sys.stdout.flush()
    return _run


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Chạy toàn bộ pipeline ALNS: load → init → optimize → 2-opt → save → visualize."""
    config = load_config()
    data   = load_data(config)

    matrix   = data['distance_matrix']
    capacity = data['vehicle_capacity']
    df_locs  = data['df_locations']
    demands  = data['demands']

    init_state   = build_initial_state(matrix, capacity, demands, config)
    init_cost_km = sum(init_state.route_cost(r) for r in init_state.routes) / KM_SCALE

    print(f"[*] {matrix.shape[0]-1} khách hàng | "
          f"{len(init_state.routes)} xe ban đầu | capacity={capacity}")
    print(f"[*] Quãng đường ban đầu: {init_cost_km:.2f} km")

    alns, accept, select, _ = configure_alns(init_state, config)

    max_no_improve = config['alns_parameters'].get('max_no_improve', 3000)
    print(f"--- Tối ưu (dừng sau {max_no_improve} vòng không cải thiện) ---")

    shared  = [init_cost_km, 0, 0]   # [best_km, unassigned, improvements]
    stop_flag = threading.Event()
    start_time = time.time()

    def on_best(state, _rnd):
        """Callback khi tìm được nghiệm tốt hơn."""
        km = sum(state.route_cost(r) for r in state.routes if len(r) > 2) / KM_SCALE
        shared[0]  = km
        shared[1]  = len(state.unassigned)
        shared[2] += 1
        sys.stdout.write(f"\n  -> [#{shared[2]}] {km:.2f} km | "
                         f"Unassigned: {shared[1]}\n")
        sys.stdout.flush()

    alns.on_best(on_best)

    t = threading.Thread(
        target=make_progress_logger(stop_flag, shared, start_time), daemon=True)
    t.start()

    result_alns = alns.iterate(init_state, select, accept,
                                stop=NoImprovementStop(max_no_improve))

    stop_flag.set()
    t.join(timeout=1)

    best_state = result_alns.best_state

    print("\n--- 2-opt làm mịn lộ trình ---")
    best_state.apply_2opt()

    elapsed = time.time() - start_time

    # Tính tổng chi phí đơn vị nội bộ rồi chuyển km trong build_result
    total_cost_units = sum(
        best_state.route_cost(r)
        for r in best_state.routes if len(r) > 2
    )

    result = build_result("ALNS", best_state.routes, total_cost_units, elapsed)

    save_result(result, config, "ALNS")
    visualize(result, config, "ALNS", df_locs)

    print(f"\n[ALNS DONE] {result['total_distance_km']:.2f} km | "
          f"{result['num_vehicles']} xe | {elapsed:.2f}s")


if __name__ == "__main__":
    main()