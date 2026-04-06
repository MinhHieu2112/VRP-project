"""
Algorithms/ALNS/main.py  (refactored)
=======================================
Entry-point cho ALNS solver. Tách khởi tạo nghiệm ra init_strategies.py.

Sửa lỗi so với phiên bản cũ:
  [FIX-1] loader.py nhân matrix * 100: mâu thuẫn với comment "đơn vị mét".
          Giờ đọc nguyên giá trị mét từ OSRM (không scale).
  [FIX-2] build_initial_solution: thay bằng gọi init_strategies.capacity_greedy_init
          (logic giống nhau nhưng tập trung một nơi, có validation).
  [FIX-3] on_best callback: tính dist trực tiếp từ route_cost (mét) rồi /1000.
"""

import numpy as np
import time
import json
import os
import sys
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Algorithms.Init_strategies.Init_strategies import init_solution, _build_demands
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer

from alns.stop import MaxIterations
from src.state import CvrpState
from src.solver import configure_alns

METERS_TO_KM = 1000


# ──────────────────────────────────────────────────────────────────────────────
# Stopping criterion
# ──────────────────────────────────────────────────────────────────────────────

class NoImprovementStop:
    """Dừng khi best solution không cải thiện sau max_no_improve vòng liên tiếp."""

    def __init__(self, max_no_improve: int):
        """Khởi tạo với ngưỡng dừng."""
        self._max_no_improve  = max_no_improve
        self._no_improve_count = 0
        self._best_obj        = float('inf')

    def __call__(self, rng, best, curr) -> bool:
        """Kiểm tra điều kiện dừng mỗi vòng lặp."""
        obj = best.objective()
        if obj < self._best_obj - 1e-6:
            self._best_obj        = obj
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1
        return self._no_improve_count >= self._max_no_improve


# ──────────────────────────────────────────────────────────────────────────────
# Config & Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str = 'config.json') -> dict:
    """Đọc file JSON cấu hình."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_distance_matrix(file_path: str) -> np.ndarray:
    """
    Đọc ma trận khoảng cách từ CSV.
    [FIX-1] Trả về nguyên giá trị mét (int), KHÔNG nhân thêm hệ số scale.
    """
    import pandas as pd
    data = pd.read_csv(file_path, header=None)
    # Clip giá trị âm nhỏ (OSRM rounding noise) về 0
    matrix = np.clip(data.values, 0, None).astype(np.int64)
    print(f"[Loader] Ma trận {matrix.shape}, min={matrix.min()}, max={matrix.max()} (mét)")
    return matrix


# ──────────────────────────────────────────────────────────────────────────────
# Progress monitoring (background thread)
# ──────────────────────────────────────────────────────────────────────────────

def make_progress_printer(stop_flag: threading.Event,
                           best_so_far: list,
                           best_unassigned: list,
                           improvement_count: list,
                           start_time: float):
    """
    Tạo hàm in tiến độ chạy trên thread riêng, in mỗi 10 giây.
    Dùng list 1 phần tử làm mutable container (thay global).
    """
    def _printer():
        while not stop_flag.is_set():
            stop_flag.wait(10)
            if stop_flag.is_set():
                break
            elapsed = time.time() - start_time
            sys.stdout.write(
                f"\r  [{elapsed:5.0f}s] Best: {best_so_far[0]:.2f} km | "
                f"Unassigned: {best_unassigned[0]} | "
                f"Cải thiện: {improvement_count[0]} lần   "
            )
            sys.stdout.flush()
    return _printer


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Chạy toàn bộ pipeline ALNS: load → init → optimize → save."""
    config = load_config()

    matrix   = load_distance_matrix(config['data_path'])
    if matrix is None:
        return

    num_nodes   = matrix.shape[0]
    constraints = config['constraints']
    max_v       = constraints['max_vehicles']
    cap         = constraints['vehicle_capacity']

    demands_map = _build_demands(num_nodes, demands=None,
                                  default_demand=constraints.get('default_demand', 1))

    # [FIX-2] Dùng init_strategies thay vì build_initial_solution nội tuyến
    clients         = list(range(1, num_nodes))
    initial_routes  = init_solution(
        strategy       = "capacity_greedy",
        matrix         = matrix,
        num_nodes      = num_nodes,
        capacity       = cap,
        demands        = demands_map,
        default_demand = 1.0,
        max_vehicles   = max_v,
        validate       = True,
        clients        = clients,
    )

    initial_state = CvrpState(initial_routes, [], matrix, cap, demands_map, config)

    init_dist_m = sum(initial_state.route_cost(r) for r in initial_routes)
    print(f"[*] {num_nodes - 1} khách hàng | {len(initial_routes)} xe ban đầu | "
          f"capacity={cap}")
    print(f"[*] Quãng đường ban đầu: {init_dist_m:.0f}m "
          f"({init_dist_m / METERS_TO_KM:.2f}km)")

    alns, accept, select, _ = configure_alns(initial_state, config)

    p              = config['alns_parameters']
    max_no_improve = p.get('max_no_improve', 3000)
    print(f"--- Bắt đầu tối ưu (dừng sau {max_no_improve} vòng không cải thiện) ---")

    # Mutable containers để callback thread-safe
    best_so_far      = [init_dist_m / METERS_TO_KM]
    best_unassigned  = [0]
    improvement_count = [0]

    def on_best(state, rnd):
        """Callback khi tìm được nghiệm tốt hơn."""
        # [FIX-3] Tính km trực tiếp từ route_cost (mét)
        dist_km = sum(state.route_cost(r)
                      for r in state.routes if len(r) > 2) / METERS_TO_KM
        best_so_far[0]         = dist_km
        best_unassigned[0]     = len(state.unassigned)
        improvement_count[0]  += 1
        sys.stdout.write(
            f"\n  -> [#{improvement_count[0]}] Cải thiện: {dist_km:.2f} km"
            f" | Unassigned: {len(state.unassigned)}\n"
        )
        sys.stdout.flush()

    alns.on_best(on_best)

    stop_flag  = threading.Event()
    start_time = time.time()

    printer_fn = make_progress_printer(stop_flag, best_so_far,
                                        best_unassigned, improvement_count,
                                        start_time)
    t = threading.Thread(target=printer_fn, daemon=True)
    t.start()

    stop   = NoImprovementStop(max_no_improve=max_no_improve)
    result = alns.iterate(initial_state, select, accept, stop=stop)

    stop_flag.set()
    t.join(timeout=1)

    best_state = result.best_state

    print("\n--- Đang làm mịn lộ trình với 2-opt ---")
    best_state.apply_2opt()

    end_time = time.time()

    actual_routes   = [r for r in best_state.routes if len(r) > 2]
    routes_dict     = {i: [int(n) for n in r]
                       for i, r in enumerate(actual_routes)}
    final_dist_km   = sum(best_state.route_cost(r)
                          for r in actual_routes) / METERS_TO_KM

    loads   = [int(sum(demands_map[n] for n in r if n != 0))
               for r in actual_routes]
    avg_load = np.mean(loads) if loads else 0
    avg_pts  = np.mean([len(r) - 2 for r in actual_routes]) if actual_routes else 0

    standardized_result = {
        "solver_name":       "ALNS",
        "total_distance_km": final_dist_km,
        "execution_time":    end_time - start_time,
        "routes":            routes_dict,
        "num_vehicles":      len(routes_dict),
    }

    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'ALNS')
    os.makedirs(output_dir, exist_ok=True)
    ResultHandler.save_to_txt(standardized_result, output_dir)

    print(f"\n[HOÀN TẤT]")
    print(f"Tổng quãng đường: {final_dist_km:.2f} km")
    print(f"Số xe sử dụng:    {len(routes_dict)}")
    print(f"Trung bình:       {avg_pts:.1f} điểm/xe | "
          f"{avg_load:.1f} tải/xe (capacity={cap})")
    print(f"Thời gian:        {end_time - start_time:.2f} giây")
    print(f"Số lần cải thiện: {improvement_count[0]}")


if __name__ == "__main__":
    main()