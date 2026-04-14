"""
Algorithms/ALNS/main.py
=======================
Entry-point cho ALNS solver.

Luồng chuẩn hóa:
  Pipeline.load_data()        → matrix (unit = 10m, KM_SCALE=100), demands, df_locs
  build_initial_state()       → CvrpState (nghiệm ban đầu capacity_greedy)
  configure_alns()            → alns, accept (SA autofit), select (RouletteWheel)
  alns.iterate(NoImprovementStop) → tối ưu
  best_state.apply_2opt()     → làm mịn
  Pipeline.build_result()     → chuẩn hóa kết quả (tự chia KM_SCALE)
  Pipeline.save_result()      → lưu txt
  Pipeline.visualize()        → bản đồ folium
"""

import os
import sys
import json
import time
import threading
import numpy as np

# Đảm bảo PROJECT_ROOT trong sys.path để import Utils.*
_THIS_DIR    = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Utils.Pipeline import load_data, build_result, save_result, visualize, KM_SCALE
from Algorithms.Init_strategies.Init_strategies import init_solution
from src.state import CvrpState
from src.solver import configure_alns

# ── Stopping criterion ────────────────────────────────────────────────────────

class NoImprovementStop:
    """Dừng khi best objective không cải thiện sau max_no_improve vòng liên tiếp."""

    def __init__(self, max_no_improve: int):
        self._limit = max_no_improve
        self._count = 0
        self._best  = float('inf')

    def __call__(self, rng, best, curr) -> bool:
        obj = best.objective()
        if obj < self._best - 1e-6:
            self._best  = obj
            self._count = 0
        else:
            self._count += 1
        return self._count >= self._limit


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str = None) -> dict:
    """Đọc config.json của ALNS (nằm cùng thư mục với main.py)."""
    if path is None:
        path = os.path.join(_THIS_DIR, 'config.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_initial_state(data: dict, config: dict) -> CvrpState:
    """
    Tạo nghiệm ban đầu bằng capacity_greedy, bọc vào CvrpState.

    data['distance_matrix'] đã được Pipeline chuẩn hóa (unit = 10m).
    demands là np.ndarray → chuyển sang dict để Init_strategies dùng.
    """
    matrix   = data['distance_matrix']
    capacity = data['vehicle_capacity']
    demands_arr = data['demands']
    num_nodes   = matrix.shape[0]

    demands_dict = {i: int(demands_arr[i]) for i in range(num_nodes)}
    constraints  = config.get('global_constraints', config.get('constraints', {}))

    routes = init_solution(
        strategy       = "clarke_wright",
        matrix         = matrix,
        num_nodes      = num_nodes,
        capacity       = capacity,
        demands        = demands_dict,
        default_demand = constraints.get('default_demand', 1),
        max_vehicles   = constraints.get('max_vehicles', 200),
        validate       = True,
    )
    return CvrpState(routes, [], matrix, capacity, demands_dict, config)


def _progress_logger(stop_flag: threading.Event, shared: list, start_time: float):
    """In tiến độ mỗi 10 giây trên background thread."""
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
    """Pipeline đầy đủ: load → init → ALNS → 2-opt → save → visualize."""

    # 1. Config & data
    config = load_config()
    data   = load_data(config)          # DataLoader: matrix unit=10m, KM_SCALE=100

    matrix   = data['distance_matrix']
    df_locs  = data['df_locations']
    capacity = data['vehicle_capacity']

    # 2. Nghiệm ban đầu
    init_state    = build_initial_state(data, config)
    init_cost_km  = sum(init_state.route_cost(r)
                        for r in init_state.routes) / KM_SCALE

    print(f"[*] {matrix.shape[0]-1} khách hàng | "
          f"{len(init_state.routes)} xe ban đầu | capacity={capacity}")
    print(f"[*] Quãng đường ban đầu: {init_cost_km:.2f} km")

    # 3. Cấu hình ALNS
    #    configure_alns dùng SA.autofit từ init_obj thực → temperature tự khớp scale
    alns, accept, select, _ = configure_alns(init_state, config)

    p              = config['alns_parameters']
    max_no_improve = p.get('max_no_improve', 2000)
    print(f"--- Tối ưu (dừng sau {max_no_improve} vòng không cải thiện) ---")

    # 4. Shared state cho callback + progress thread
    shared     = [init_cost_km, 0, 0]   # [best_km, unassigned, improvements]
    stop_flag  = threading.Event()
    start_time = time.time()

    def on_best(state, _rnd):
        km = sum(state.route_cost(r)
                 for r in state.routes if len(r) > 2) / KM_SCALE
        shared[0]  = km
        shared[1]  = len(state.unassigned)
        shared[2] += 1
        sys.stdout.write(
            f"\n  -> [#{shared[2]}] {km:.2f} km | Unassigned: {shared[1]}\n"
        )
        sys.stdout.flush()

    alns.on_best(on_best)

    t = threading.Thread(
        target=_progress_logger(stop_flag, shared, start_time), daemon=True
    )
    t.start()

    # 5. Chạy ALNS
    result_alns = alns.iterate(
        init_state, select, accept,
        stop=NoImprovementStop(max_no_improve)
    )

    stop_flag.set()
    t.join(timeout=1)

    # 6. 2-opt làm mịn
    best_state = result_alns.best_state
    print("\n--- 2-opt làm mịn lộ trình ---")
    best_state.apply_2opt()

    elapsed = time.time() - start_time

    # 7. Build & save result
    #    build_result nhận total_cost_units (đơn vị nội bộ) → tự chia KM_SCALE
    active_routes    = [r for r in best_state.routes if len(r) > 2]
    total_cost_units = sum(best_state.route_cost(r) for r in active_routes)

    result = build_result("ALNS", active_routes, total_cost_units, elapsed)

    save_result(result, config, "ALNS")
    visualize(result, config, "ALNS", df_locs)

    print(f"\n[ALNS DONE] {result['total_distance_km']:.2f} km | "
          f"{result['num_vehicles']} xe | {elapsed:.2f}s")


if __name__ == "__main__":
    main()