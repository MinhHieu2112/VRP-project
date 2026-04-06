"""
Algorithms/MILP/main.py  (refactored)
=======================================
Entry-point cho MILP solver (PuLP / CBC).

Sửa lỗi so với phiên bản cũ:
  [FIX-1] Khởi tạo nghiệm ban đầu (upper bound) dùng init_strategies.
  [FIX-2] Depot_Flow_In constraint đã đúng (= total_demand) — giữ nguyên.
  [FIX-3] _extract_routes: giới hạn max_steps để tránh infinite loop
          khi CBC trả về nghiệm xấp xỉ (timelimit hit).
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from milp_solvers import solve_acvrp_milp

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

sys.path.append(PROJECT_ROOT)
from Algorithms.Init_strategies.Init_strategies import greedy_init, _build_demands
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer

METERS_TO_KM = 1000


# ──────────────────────────────────────────────────────────────────────────────
# Config & Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Đọc file JSON cấu hình MILP."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_and_prep_matrix(matrix_path: str, limit_nodes: int) -> np.ndarray:
    """
    Đọc ma trận khoảng cách OSRM (mét, int), cắt theo limit_nodes.
    Clip giá trị âm nhỏ (floating-point noise từ OSRM) về 0.
    """
    df = pd.read_csv(matrix_path, header=None)
    mat = df.values.astype(np.int64)

    # Reshape nếu CSV flatten thành 1 dòng
    if mat.shape[0] == 1:
        n_total = int(np.sqrt(mat.size))
        mat = mat.reshape((n_total, n_total))

    n   = min(limit_nodes, len(mat))
    mat = np.clip(mat[:n, :n], 0, None)
    print(f"[MILP] Sử dụng {n}/{len(df)} node. Ma trận: mét (int).")
    return mat


def build_demands(config: dict, num_nodes: int) -> dict:
    """
    Xây dựng dict demands từ config.
    Hỗ trợ demands dạng int (uniform) hoặc list (per-node).
    """
    raw = config.get('demands', 1)
    customers = list(range(1, num_nodes))
    demands = {0: 0}

    if isinstance(raw, int):
        for i in customers:
            demands[i] = raw
    elif isinstance(raw, list):
        if len(raw) < num_nodes:
            raise ValueError(
                f"Danh sách demands ({len(raw)}) ngắn hơn số node ({num_nodes})."
            )
        for i in customers:
            demands[i] = raw[i]
    else:
        raise TypeError("'demands' trong config phải là int hoặc list.")

    return demands


# ──────────────────────────────────────────────────────────────────────────────
# Warmstart — cung cấp upper bound cho CBC
# ──────────────────────────────────────────────────────────────────────────────

def compute_upper_bound(matrix: np.ndarray, demands: dict,
                         capacity: float, max_vehicles: int) -> float:
    """
    [FIX-1] Tính upper bound bằng NNH để CBC có điểm xuất phát tốt hơn.
    PuLP CBC không hỗ trợ warm-start trực tiếp, nhưng việc in UB
    giúp chọn timelimit phù hợp.
    """
    num_nodes = matrix.shape[0]
    solution  = greedy_init(
        matrix        = matrix,
        num_nodes     = num_nodes,
        capacity      = capacity,
        demands       = demands,
        max_vehicles  = max_vehicles,
    )
    ub_m = sum(
        sum(matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
        for route in solution
    )
    print(f"[MILP] Upper bound (NNH): {ub_m:.0f}m ({ub_m / METERS_TO_KM:.2f}km) | "
          f"{len(solution)} xe")
    return float(ub_m)


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def format_routes(routes_info: list) -> dict:
    """Chuyển routes_info thành dict {vehicle_id: route_list}."""
    routes_dict = {}
    for idx, info in enumerate(routes_info):
        route = info['route']
        if route[-1] != 0:
            route.append(0)
        routes_dict[idx] = route
    return routes_dict


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(limit_nodes: int = 50):
    """Chạy toàn bộ pipeline MILP: load → upper bound → solve → save."""
    config_path = os.path.join(CURRENT_DIR, 'config.json')
    matrix_path = os.path.join(PROJECT_ROOT, "Data", "orsm_matrix.csv")
    loc_path    = os.path.join(PROJECT_ROOT, "Data", "locations.csv")

    config  = load_config(config_path)
    matrix  = load_and_prep_matrix(matrix_path, limit_nodes)
    n       = matrix.shape[0]
    demands = build_demands(config, n)

    Q         = config.get('vehicle_capacity', 10)
    K         = config.get('num_vehicles', 200)
    timelimit = config.get('max_runtime_seconds', 120)

    print(f"[MILP] {n} node | {K} xe | Capacity={Q} | Timelimit={timelimit}s")
    print(f"[MILP] Đơn vị ma trận: mét. Kết quả báo cáo: km (÷1000)")

    # [FIX-1] Upper bound
    compute_upper_bound(matrix, demands, Q, K)

    print("--- ĐANG GIẢI BẰNG MILP (PuLP/CBC) ---")
    start_time = time.time()
    status_str, obj_val_m, routes_info = solve_acvrp_milp(
        matrix, demands,
        num_vehicles=K,
        capacity=Q,
        timelimit=timelimit
    )
    duration = time.time() - start_time

    print(f"\n{'='*50}")
    print(f"TRẠNG THÁI SOLVER: {status_str}")

    if obj_val_m is None:
        print("[KẾT QUẢ] Solver không tìm được nghiệm khả thi.")
        print("="*50)
        return

    routes_dict = format_routes(routes_info)
    total_km    = obj_val_m / METERS_TO_KM

    standardized_result = {
        "solver_name":       "MILP",
        "total_distance_km": total_km,
        "execution_time":    duration,
        "routes":            routes_dict,
        "num_vehicles":      len(routes_dict),
    }

    print(f"TỔNG KHOẢNG CÁCH: {total_km:.2f} km")
    print(f"Số xe sử dụng:    {standardized_result['num_vehicles']}")
    print(f"Thời gian chạy:   {duration:.2f} s")

    invalid_routes = [i for i, info in enumerate(routes_info)
                      if not info['is_valid']]
    if invalid_routes:
        print(f"[CẢNH BÁO] {len(invalid_routes)} tuyến vi phạm capacity: "
              f"{invalid_routes}")
    print("=" * 50)

    output_dir = os.path.join(PROJECT_ROOT, "Results", "MILP")
    os.makedirs(output_dir, exist_ok=True)
    ResultHandler.save_to_txt(standardized_result, output_dir)

    try:
        if os.path.exists(loc_path):
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
    main(limit_nodes=50)