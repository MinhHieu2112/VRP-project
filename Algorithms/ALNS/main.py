import numpy as np
import time
import json
import os
import sys
import pandas as pd
import math
import threading
from alns.stop import MaxIterations

# Import local modules
from src.utils.loader import load_distance_matrix
from src.state import CvrpState
from src.solver import configure_alns

# Import Project Utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer


def load_config(path='config.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    config = load_config()
    matrix = load_distance_matrix(config['data_path'])
    if matrix is None:
        return

    num_nodes = matrix.shape[0]
    scaling = config.get('common_model_parameters', {}).get('scaling_factor', 100)

    # Khởi tạo dữ liệu
    demands = np.ones(num_nodes) * config['constraints'].get('default_demand', 1)
    demands[0] = 0

    # Tạo nghiệm ban đầu (Mỗi xe 1 cụm khách)
    clients = list(range(1, num_nodes))
    max_v = config['constraints']['max_vehicles']
    cap = config['constraints']['vehicle_capacity']

    initial_routes = []
    chunk = int(np.ceil(len(clients) / max_v))
    for i in range(0, len(clients), chunk):
        initial_routes.append([0] + clients[i:i + chunk] + [0])

    initial_state = CvrpState(initial_routes, [], matrix, cap, demands, config)

    # 1. Cấu hình ALNS
    alns, accept, select, _ = configure_alns(initial_state, config)

    # 2. Tính toán số vòng lặp hội tụ
    p = config['alns_parameters']
    iters = int(math.log(p['end_temperature'] / p['start_temperature']) / math.log(p['step'])) + 500

    print(f"Quãng đường ban đầu: {initial_state.objective() / scaling:.2f} km")
    print(f"--- Bắt đầu tối ưu ({iters} vòng lặp) ---")

    # --- Progress logging ---
    best_so_far = [sum(initial_state.route_cost(r) for r in initial_state.routes if len(r) > 2) / scaling]
    best_unassigned = [len(initial_state.unassigned)]
    improvement_count = [0]

    def on_best(state, rnd):
        dist = sum(state.route_cost(r) for r in state.routes if len(r) > 2) / scaling
        best_so_far[0] = dist
        best_unassigned[0] = len(state.unassigned)
        improvement_count[0] += 1
        sys.stdout.write(
            f"\n  -> [#{improvement_count[0]}] Cải thiện: {dist:.2f} km"
            f" | Unassigned: {len(state.unassigned)}\n"
        )
        sys.stdout.flush()

    alns.on_best(on_best)

    # Thread in tiến độ mỗi 10 giây
    stop_flag = threading.Event()
    start_time = time.time()

    def progress_printer():
        while not stop_flag.is_set():
            stop_flag.wait(10)
            if stop_flag.is_set():
                break
            elapsed = time.time() - start_time
            sys.stdout.write(
                f"\r  [{elapsed:5.0f}s] Đang tối ưu... "
                f"Best: {best_so_far[0]:.2f} km | "
                f"Unassigned: {best_unassigned[0]} | "
                f"Cải thiện: {improvement_count[0]} lần   "
            )
            sys.stdout.flush()

    t = threading.Thread(target=progress_printer, daemon=True)
    t.start()

    # ── DEBUG: Thử 1 vòng destroy + repair để kiểm tra SA temperature ──
    from src.operators.destroy_operators import random_removal
    from src.operators.repair_operators import greedy_insertion
    _rng = np.random.RandomState(42)
    _destroyed = random_removal(initial_state, _rng)
    _repaired  = greedy_insertion(_destroyed, _rng)
    _delta = _repaired.objective() - initial_state.objective()
    _accept_prob = min(1.0, math.exp(-_delta / p['start_temperature'])) if _delta > 0 else 1.0
    print(f"\n[DEBUG] objective ban đầu:       {initial_state.objective():.2f}")
    print(f"[DEBUG] objective sau destroy:    {_destroyed.objective():.2f}")
    print(f"[DEBUG] objective sau repair:     {_repaired.objective():.2f}")
    print(f"[DEBUG] unassigned sau destroy:   {len(_destroyed.unassigned)}")
    print(f"[DEBUG] unassigned sau repair:    {len(_repaired.unassigned)}")
    print(f"[DEBUG] scaling_factor:           {scaling}")
    print(f"[DEBUG] start_temperature:        {p['start_temperature']}")
    print(f"[DEBUG] delta objective:          {_delta:.2f}")
    print(f"[DEBUG] SA accept prob (delta>0): {_accept_prob:.6f}")
    print(f"[DEBUG] penalty_unassigned:       {config['constraints'].get('penalty_unassigned', 'N/A')}")
    print()
    # ── END DEBUG ──

    result = alns.iterate(initial_state, select, accept, stop=MaxIterations(iters))

    stop_flag.set()
    t.join(timeout=1)

    best_state = result.best_state

    # 3. Local Search tinh chỉnh cuối cùng
    print("\n--- Đang làm mịn lộ trình với 2-opt ---")
    best_state.apply_2opt()

    end_time = time.time()

    # 4. Xuất kết quả
    actual_routes = [r for r in best_state.routes if len(r) > 2]
    routes_dict = {i: [int(node) for node in r] for i, r in enumerate(actual_routes)}

    final_dist = sum(best_state.route_cost(r) for r in actual_routes) / scaling

    standardized_result = {
        "solver_name": "ALNS_Full_Optimized",
        "total_distance_km": final_dist,
        "execution_time": end_time - start_time,
        "routes": routes_dict,
        "num_vehicles": len(routes_dict)
    }

    # Lưu và In
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'ALNS')
    os.makedirs(output_dir, exist_ok=True)
    ResultHandler.save_to_txt(standardized_result, output_dir)

    print(f"\n[HOÀN TẤT]")
    print(f"Tổng quãng đường: {final_dist:.2f} km")
    print(f"Số xe sử dụng:    {len(routes_dict)}")
    print(f"Thời gian:        {end_time - start_time:.2f} giây")
    print(f"Số lần cải thiện: {improvement_count[0]}")


if __name__ == "__main__":
    main()