import os
import sys
import json
import time
import pandas as pd
import numpy as np
from milp_solvers import solve_acvrp_milp

# ===== XÁC ĐỊNH ĐƯỜNG DẪN =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# Import Utils chung từ project root
sys.path.append(PROJECT_ROOT)
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer

def load_and_prep_data(matrix_path, config_path, limit_nodes):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {config_path}")
        return None, None, None

    df = pd.read_csv(matrix_path, header=None)
    matrix_full = df.values

    if matrix_full.shape[0] == 1:
        n_total = int(np.sqrt(matrix_full.size))
        matrix_full = matrix_full.reshape((n_total, n_total))

    n = min(limit_nodes, len(matrix_full))
    print(f"[*] Đang sử dụng {n} điểm đầu tiên trong ma trận dữ liệu làm Input.")
    matrix = matrix_full[:n, :n]

    nodes = list(range(n))
    customers = list(range(1, n))

    raw_demands = config.get('demands', 1)

    demands = {0: 0}
    if isinstance(raw_demands, int):
        for i in customers:
            demands[i] = raw_demands
    elif isinstance(raw_demands, list):
        if len(raw_demands) < n:
            raise ValueError(f"Danh sách demands ({len(raw_demands)}) ngắn hơn số điểm đang giải ({n}).")
        for i in customers:
            demands[i] = raw_demands[i]
    else:
        raise TypeError("Demands trong config phải là số nguyên hoặc mảng/list.")

    return matrix, demands, config

if __name__ == "__main__":
    matrix_path = os.path.join(CURRENT_DIR, 'orsm_matrix_scaled.csv')
    config_path = os.path.join(CURRENT_DIR, 'config.json')
    limit_nodes = 350

    # Đọc locations cho visualization
    loc_path = os.path.join(PROJECT_ROOT, "Data", "locations.csv")

    print("--- BẮT ĐẦU ĐỌC DỮ LIỆU ---")
    matrix, demands, config = load_and_prep_data(matrix_path, config_path, limit_nodes)

    if matrix is not None:
        Q = config.get('vehicle_capacity', 10)
        K = config.get('num_vehicles', 10)
        timelimit = config.get('max_runtime_seconds', 120)

        print(f"[*] Quy mô: {len(matrix)} điểm | {K} xe | Sức chứa: {Q}")
        print(f"--- ĐANG GIẢI BẰNG MILP (Giới hạn: {timelimit}s) ---")

        start_time = time.time()
        status_str, obj_val, routes_info = solve_acvrp_milp(
            matrix, demands, num_vehicles=K, capacity=Q, timelimit=timelimit
        )
        duration = time.time() - start_time

        print("\n" + "="*50)
        print(f"TRẠNG THÁI: {status_str}")

        if obj_val is not None:
            # === TẠO KẾT QUẢ CHUẨN ===
            routes_dict = {}
            for idx, info in enumerate(routes_info):
                route = info['route']
                # Đảm bảo route kết thúc bằng depot
                if route[-1] != 0:
                    route.append(0)
                routes_dict[idx] = route

            standardized_result = {
                "solver_name": "MILP",
                "total_distance_km": obj_val / 100,  # Chia scaling_factor
                "execution_time": duration,
                "routes": routes_dict,
                "num_vehicles": len(routes_dict)
            }

            # === IN KẾT QUẢ ===
            print(f"TỔNG KHOẢNG CÁCH: {standardized_result['total_distance_km']:.2f} km")
            print(f"Số xe sử dụng: {standardized_result['num_vehicles']}")
            print(f"Thời gian chạy: {duration:.2f} s")

            # === LƯU KẾT QUẢ BẰNG RESULTHANDLER CHUNG ===
            output_dir = os.path.join(PROJECT_ROOT, "Results", "MILP")
            os.makedirs(output_dir, exist_ok=True)

            ResultHandler.save_to_txt(standardized_result, output_dir)
            ResultHandler.save_to_json(standardized_result, output_dir)

            # === TRỰC QUAN HÓA BẰNG VISUALIZER CHUNG ===
            print("--- Đang khởi tạo bản đồ trực quan ---")
            try:
                if os.path.exists(loc_path):
                    df_locations = pd.read_csv(loc_path)
                    vis = Visualizer(
                        df_locations,
                        osrm_url="http://localhost:5001",
                        use_osrm=True
                    )
                    map_path = os.path.join(output_dir, "route_map.html")
                    vis.draw(standardized_result['routes'], map_path)
                    print(f"[HOÀN TẤT] Bản đồ lưu tại: {map_path}")
                else:
                    print(f"Thiếu file locations: {loc_path}")
            except Exception as e:
                print(f"[WARNING] Trực quan hóa thất bại: {e}")

        else:
            print("Solver không tìm thấy nghiệm nguyên khả thi nào.")
        print("="*50)
