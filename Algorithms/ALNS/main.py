# File chạy chính của thuật toán ALNS phối hợp luồng giải để tối ưu hóa lộ trình VRP.
import os
import sys
import json
import time
import threading
import numpy as np
from typing import Optional

_THIS_DIR     = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Utils.Pipeline import load_data, build_result, save_result, visualize, KM_SCALE
from Algorithms.Init_strategies.Init_strategies import init_solution
from src.state import CvrpState
from src.solver import configure_alns

class NoImprovementStop:
    # Điều kiện dừng khi hàm mục tiêu không cải thiện sau số vòng lặp tối đa.

    def __init__(self, max_no_improve: int):
        # Khởi tạo giới hạn số vòng lặp và bộ đếm không cải thiện.
        self._limit = max_no_improve
        self._count = 0
        self._best  = float('inf')

    def __call__(self, rng, best, curr) -> bool:
        # Kiểm tra điều kiện dừng dựa trên cải thiện của hàm mục tiêu tốt nhất.
        obj = best.objective()
        if obj < self._best - 1e-6:
            self._best  = obj
            self._count = 0
        else:
            self._count += 1
        return self._count >= self._limit


def load_config(path: Optional[str] = None) -> dict:
    # Tải cấu hình tham số của thuật toán từ tệp JSON.
    if path is None:
        path = os.path.join(_THIS_DIR, 'config.json')
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    if not isinstance(config, dict):
        config = {}
    for key in ['global_constraints', 'constraints', 'alns_parameters']:
        if config.get(key) is None:
            config[key] = {}
    return config


def build_initial_state(data: dict, config: dict) -> CvrpState:
    # Khởi tạo trạng thái nghiệm ban đầu cho mô hình VRP đồng thời dựng candidate list.
    matrix   = data['distance_matrix']
    capacity = data['vehicle_capacity']
    demands_arr = data['demands']
    num_nodes   = matrix.shape[0]

    demands_dict = {i: int(demands_arr[i]) for i in range(num_nodes)}
    constraints  = config.get('global_constraints') or config.get('constraints') or {}

    from Algorithms.Tabu.utils import build_granular_lists
    alns_cfg      = config.get('alns_parameters') or {}
    granular_beta = alns_cfg.get('granular_beta', 1.5)
    granular_k    = alns_cfg.get('granular_k', 20)
    print(f"[ALNS] Xây dựng candidate list granular: β={granular_beta}, k={granular_k}")
    config["granular_neighbors"] = build_granular_lists(matrix, num_nodes, granular_beta, granular_k)

    init_strategy = alns_cfg.get('init_strategy', 'clarke_wright')
    print(f"[ALNS] Khởi tạo nghiệm bằng chiến lược: '{init_strategy}'")

    routes = init_solution(
        strategy       = init_strategy,
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
    # In thông tin tiến độ tối ưu hóa ra màn hình console định kỳ.
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


def main():
    # Điều phối quy trình tối ưu VRP bằng ALNS từ nạp dữ liệu đến lưu kết quả.
    config = load_config()
    data   = load_data(config)

    matrix   = data['distance_matrix']
    df_locs  = data['df_locations']
    capacity = data['vehicle_capacity']

    init_state   = build_initial_state(data, config)
    init_cost_km = sum(init_state.route_costs) / KM_SCALE

    print(f"[*] {matrix.shape[0]-1} khách hàng | "
          f"{len(init_state.routes)} xe ban đầu | capacity={capacity}")
    print(f"[*] Quãng đường ban đầu: {init_cost_km:.2f} km")

    alns, accept, select, _ = configure_alns(init_state, config)

    p              = config['alns_parameters']
    max_no_improve = p.get('max_no_improve', 2000)
    print(f"--- Tối ưu (dừng sau {max_no_improve} vòng không cải thiện) ---")

    shared     = [init_cost_km, 0, 0]
    stop_flag  = threading.Event()
    start_time = time.time()

    def on_best(state, _rnd):
        # Cập nhật kết quả tốt nhất khi tìm thấy lời giải cải tiến.
        km = sum(state.route_costs) / KM_SCALE
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

    result_alns = alns.iterate(
        init_state, select, accept,
        stop=NoImprovementStop(max_no_improve)
    )

    stop_flag.set()
    t.join(timeout=1)

    best_state = result_alns.best_state
    print("\n--- 2-opt làm mịn lộ trình ---")
    best_state.apply_2opt()

    elapsed = time.time() - start_time

    active_routes    = [r for r in best_state.routes if len(r) > 2]
    total_cost_units = sum(best_state.route_costs)

    result = build_result("ALNS", active_routes, total_cost_units, elapsed)

    save_result(result, config, "ALNS")
    visualize(result, config, "ALNS", df_locs)

    print(f"\n[ALNS DONE] {result['total_distance_km']:.2f} km | "
          f"{result['num_vehicles']} xe | {elapsed:.2f}s")


if __name__ == "__main__":
    main()