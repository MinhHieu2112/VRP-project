"""
Algorithms/Tabu/main_tabu.py  — FIX ACVRP
==========================================
[FIX-TABU-1] init_solution cũ không đảm bảo mỗi KH thăm đúng 1 lần:
  - Tạo sẵn 200 route rỗng [[0,0], [0,0], ...]
  - Shuffle khách rồi gán tuần tự → nếu gán fail (demand > capacity cho xe cuối)
    in cảnh báo nhưng bỏ qua → khách đó không được phục vụ.
  - Vi phạm ràng buộc 1 (mỗi KH thăm đúng 1 lần).

Fix: dùng greedy_init từ init_strategies (NNH đảm bảo cover tất cả KH).
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

sys.path.append(PROJECT_ROOT)

# [FIX-TABU-1] Import init_strategies thay vì dùng init_solution cũ
from Algorithms.Init_strategies.Init_strategies import greedy_init, _build_demands
from Algorithms.Tabu.tabu_solver import TabuSearchSolver
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer

METERS_TO_KM = 1000


def load_config(config_path: str) -> dict:
    """Đọc file JSON cấu hình Tabu Search."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_matrix(data_path: str) -> np.ndarray:
    """Đọc ma trận khoảng cách từ CSV (đơn vị mét, int)."""
    df  = pd.read_csv(data_path, header=None)
    mat = np.clip(df.values, 0, None).astype(np.int64)
    print(f"[Tabu] Ma trận {mat.shape}, dtype={mat.dtype}")
    return mat


def build_demands(num_nodes: int, constraints: dict) -> dict:
    """Xây dựng dict demands từ config."""
    raw = constraints.get('demands', constraints.get('default_demand', 1))
    return _build_demands(
        num_nodes,
        demands       = None,
        default_demand= float(raw) if isinstance(raw, (int, float)) else 1.0
    )


def main():
    """Chạy pipeline Tabu Search: load → init → solve → save."""
    config_path = os.path.join(CURRENT_DIR, 'config_tabu.json')
    config      = load_config(config_path)

    data_path = os.path.normpath(
        os.path.join(CURRENT_DIR, config['data_path'])
    )
    if not os.path.exists(data_path):
        data_path = os.path.join(PROJECT_ROOT, "Data", "orsm_matrix.csv")

    matrix    = load_matrix(data_path)
    num_nodes = matrix.shape[0]
    print(f"[Tabu] Đã đọc ma trận {num_nodes}×{num_nodes}")

    constraints = config['constraints']
    tabu_params = config['tabu_parameters']

    cap         = constraints['vehicle_capacity']
    max_v       = constraints['max_vehicles']
    demands_map = build_demands(num_nodes, constraints)

    # [FIX-TABU-1] Dùng greedy_init (NNH) thay vì init_solution cũ.
    # NNH đảm bảo TẤT CẢ khách hàng được phục vụ đúng 1 lần.
    print(f"[Tabu] Khởi tạo nghiệm bằng Greedy NNH...")
    initial_state = greedy_init(
        matrix        = matrix,
        num_nodes     = num_nodes,
        capacity      = cap,
        demands       = demands_map,
        default_demand= 1.0,
        max_vehicles  = max_v,
    )

    # Kiểm tra coverage
    served = {n for r in initial_state for n in r if n != 0}
    missing = set(range(1, num_nodes)) - served
    if missing:
        print(f"[WARN] {len(missing)} KH chưa được phục vụ sau init!")
    else:
        print(f"[OK] Init: {len(initial_state)} xe, "
              f"tất cả {num_nodes-1} KH được phục vụ")

    max_no_improve = tabu_params.get('max_no_improve', 1000)
    max_iterations = tabu_params.get('max_iterations', 50_000)

    print(f"\n--- Tabu Search | Dừng sau {max_no_improve} vòng không cải thiện "
          f"(tối đa {max_iterations} vòng) ---")

    solver = TabuSearchSolver(
        distance_matrix = matrix,
        demands         = demands_map,
        capacity        = cap,
        max_v           = max_v,
        tabu_size       = tabu_params['tabu_size'],
        max_iter        = max_iterations,
        max_no_improve  = max_no_improve,
    )

    start_time            = time.time()
    best_state, best_dist = solver.solve(initial_state)
    duration              = time.time() - start_time

    scaling_factor = config.get('common_model_parameters', {}).get(
        'scaling_factor', METERS_TO_KM
    )

    routes_dict = {
        idx: route
        for idx, route in enumerate(r for r in best_state if len(r) > 2)
    }

    standardized_result = {
        "solver_name":       "Tabu Search",
        "total_distance_km": best_dist / scaling_factor,
        "execution_time":    duration,
        "routes":            routes_dict,
        "num_vehicles":      len(routes_dict),
    }

    print(f"\n{'='*50}")
    print(f"Tổng quãng đường: {standardized_result['total_distance_km']:.2f} km")
    print(f"Số xe sử dụng:    {standardized_result['num_vehicles']}")
    print(f"Thời gian chạy:   {duration:.2f} giây")
    print('='*50)

    output_dir = os.path.join(PROJECT_ROOT, "Results", "Tabu")
    os.makedirs(output_dir, exist_ok=True)
    ResultHandler.save_to_txt(standardized_result, output_dir)

    loc_path = os.path.normpath(
        os.path.join(CURRENT_DIR,
                     config.get('locations_path', '../../Data/locations.csv'))
    )
    if not os.path.exists(loc_path):
        loc_path = os.path.join(PROJECT_ROOT, "Data", "locations.csv")

    try:
        df_locs = pd.read_csv(loc_path)
        vis = Visualizer(df_locs,
                         osrm_url="http://localhost:5001",
                         use_osrm=True)
        map_path = os.path.join(output_dir, "route_map.html")
        vis.draw(standardized_result['routes'], map_path)
        print(f"[HOÀN TẤT] Bản đồ lưu tại: {map_path}")
    except Exception as e:
        print(f"[WARNING] Trực quan hóa thất bại: {e}")


if __name__ == "__main__":
    main()