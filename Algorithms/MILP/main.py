import os
import sys
import json
import time
import pandas as pd
import numpy as np
from milp_solvers import solve_acvrp_milp

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

sys.path.append(PROJECT_ROOT)
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer

# Ma trận OSRM: đơn vị mét, số nguyên đã làm tròn
# Chuyển sang km CHỈ khi xuất báo cáo
METERS_TO_KM = 1000


def load_and_prep_data(matrix_path, config_path, limit_nodes):
    """
    Đọc ma trận khoảng cách (đơn vị mét, int) và cấu hình.

    Lưu ý: MILP rất tốn tài nguyên — với 1600 điểm và O(n²) biến,
    cần hardware mạnh hoặc giảm limit_nodes. Mặc định 350 để chạy được.
    Để so sánh công bằng với các thuật toán khác, cần chạy full 1600.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
        return None, None, None

    df = pd.read_csv(matrix_path, header=None)
    matrix_full = df.values.astype(np.int64)  # đơn vị mét, đã int

    if matrix_full.shape[0] == 1:
        n_total = int(np.sqrt(matrix_full.size))
        matrix_full = matrix_full.reshape((n_total, n_total))

    n = min(limit_nodes, len(matrix_full))
    print(f"[*] Sử dụng {n}/{len(matrix_full)} node. Ma trận: mét (int).")
    matrix = matrix_full[:n, :n]

    customers = list(range(1, n))
    raw_demands = config.get('demands', 1)
    demands = {0: 0}

    if isinstance(raw_demands, int):
        for i in customers:
            demands[i] = raw_demands
    elif isinstance(raw_demands, list):
        if len(raw_demands) < n:
            raise ValueError(f"Danh sách demands ({len(raw_demands)}) ngắn hơn số node ({n}).")
        for i in customers:
            demands[i] = raw_demands[i]
    else:
        raise TypeError("'demands' phải là int hoặc list.")

    return matrix, demands, config


if __name__ == "__main__":
    matrix_path = os.path.join(PROJECT_ROOT, "Data", "osrm_matrix.csv")
    config_path  = os.path.join(CURRENT_DIR, 'config.json')
    loc_path     = os.path.join(PROJECT_ROOT, "Data", "locations.csv")

    # MILP phức tạp O(n²) — giới hạn node để chạy được trong thời gian hợp lý
    # Tăng limit_nodes nếu có phần cứng mạnh hơn
    limit_nodes = 350

    print("--- BẮT ĐẦU ĐỌC DỮ LIỆU ---")
    matrix, demands, config = load_and_prep_data(matrix_path, config_path, limit_nodes)

    if matrix is None:
        print("[LỖI] Không thể đọc dữ liệu. Thoát.")
        exit(1)

    Q         = config.get('vehicle_capacity', 10)
    K         = config.get('num_vehicles', 200)
    timelimit = config.get('max_runtime_seconds', 120)

    print(f"[*] Quy mô: {len(matrix)} node | {K} xe | Capacity: {Q} | Timelimit: {timelimit}s")
    print(f"[*] Đơn vị ma trận: mét (int). Kết quả báo cáo: km (÷1000)")
    print("--- ĐANG GIẢI BẰNG MILP ---")

    start_time = time.time()
    status_str, obj_val_m, routes_info = solve_acvrp_milp(
        matrix, demands,
        num_vehicles=K,
        capacity=Q,
        timelimit=timelimit
    )
    duration = time.time() - start_time

    print("\n" + "=" * 50)
    print(f"TRẠNG THÁI SOLVER: {status_str}")

    if obj_val_m is None:
        print("[KẾT QUẢ] Solver không tìm được nghiệm khả thi.")
        print("=" * 50)
        exit(0)

    routes_dict = {}
    for idx, info in enumerate(routes_info):
        route = info['route']
        if route[-1] != 0:
            route.append(0)
        routes_dict[idx] = route

    # Quy đổi sang km CHỈ ở đây khi tạo báo cáo
    total_km = obj_val_m / METERS_TO_KM

    standardized_result = {
        "solver_name": "MILP",
        "total_distance_km": total_km,
        "execution_time": duration,
        "routes": routes_dict,
        "num_vehicles": len(routes_dict)
    }

    print(f"TỔNG KHOẢNG CÁCH: {total_km:.2f} km")
    print(f"Số xe sử dụng:    {standardized_result['num_vehicles']}")
    print(f"Thời gian chạy:   {duration:.2f} s")

    invalid_routes = [i for i, info in enumerate(routes_info) if not info['is_valid']]
    if invalid_routes:
        print(f"[CẢNH BÁO] {len(invalid_routes)} tuyến vi phạm capacity: {invalid_routes}")

    print("=" * 50)

    output_dir = os.path.join(PROJECT_ROOT, "Results", "MILP")
    os.makedirs(output_dir, exist_ok=True)
    ResultHandler.save_to_txt(standardized_result, output_dir)

    print("--- Đang khởi tạo bản đồ trực quan ---")
    try:
        if not os.path.exists(loc_path):
            print(f"[WARNING] Không tìm thấy file locations: {loc_path}")
        else:
            df_locations = pd.read_csv(loc_path)
            vis = Visualizer(df_locations, osrm_url="http://localhost:5001", use_osrm=True)
            map_path = os.path.join(output_dir, "route_map.html")
            vis.draw(standardized_result['routes'], map_path)
            print(f"[HOÀN TẤT] Bản đồ lưu tại: {map_path}")
    except Exception as e:
        print(f"[WARNING] Trực quan hóa thất bại: {e}")
