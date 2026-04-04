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

sys.path.append(PROJECT_ROOT)
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer


def load_and_prep_data(matrix_path, config_path, limit_nodes):
    """
    Đọc ma trận khoảng cách và cấu hình bài toán.

    Xử lý 2 định dạng CSV:
    - Ma trận n×n thông thường: đọc trực tiếp
    - Ma trận 1×(n²) (flatten): reshape lại thành n×n

    FIX: Thêm đọc scaling_factor từ config để dùng nhất quán khi báo cáo
    kết quả (tránh hard-code /100 ở main).

    Args:
        matrix_path:  Đường dẫn file CSV chứa ma trận khoảng cách
        config_path:  Đường dẫn file JSON cấu hình
        limit_nodes:  Giới hạn số node để giải (bài toán MILP tốn tài nguyên)

    Returns:
        (matrix, demands, config) hoặc (None, None, None) nếu lỗi
    """
    # Đọc config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
        return None, None, None

    # Đọc ma trận khoảng cách
    df = pd.read_csv(matrix_path, header=None)
    matrix_full = df.values

    # Xử lý ma trận 1D (flatten) → 2D
    if matrix_full.shape[0] == 1:
        n_total = int(np.sqrt(matrix_full.size))
        matrix_full = matrix_full.reshape((n_total, n_total))

    # Giới hạn số node
    n = min(limit_nodes, len(matrix_full))
    print(f"[*] Sử dụng {n}/{len(matrix_full)} node đầu tiên.")
    matrix = matrix_full[:n, :n]

    # Xây dựng dict demands
    customers = list(range(1, n))
    raw_demands = config.get('demands', 1)
    demands = {0: 0}

    if isinstance(raw_demands, int):
        # Demand đồng nhất cho tất cả khách hàng
        for i in customers:
            demands[i] = raw_demands
    elif isinstance(raw_demands, list):
        if len(raw_demands) < n:
            raise ValueError(
                f"Danh sách demands ({len(raw_demands)}) ngắn hơn số node ({n})."
            )
        for i in customers:
            demands[i] = raw_demands[i]
    else:
        raise TypeError("'demands' trong config phải là int hoặc list.")

    return matrix, demands, config


if __name__ == "__main__":
    """
    Entry point chính:
    Đọc dữ liệu → Giải MILP → Lưu & vẽ kết quả.
    """
    matrix_path = os.path.join(CURRENT_DIR, 'orsm_matrix_scaled.csv')
    config_path  = os.path.join(CURRENT_DIR, 'config.json')
    loc_path     = os.path.join(PROJECT_ROOT, "Data", "locations.csv")
    limit_nodes  = 350  # MILP có độ phức tạp cao, giới hạn để chạy được

    print("--- BẮT ĐẦU ĐỌC DỮ LIỆU ---")
    matrix, demands, config = load_and_prep_data(matrix_path, config_path, limit_nodes)

    if matrix is None:
        print("[LỖI] Không thể đọc dữ liệu. Thoát.")
        exit(1)

    Q          = config.get('vehicle_capacity', 10)
    K          = config.get('num_vehicles', 10)
    timelimit  = config.get('max_runtime_seconds', 120)

    # FIX: Đọc scaling_factor từ config thay vì hard-code /100
    # Nếu ma trận đã nhân 100 (để làm số nguyên cho solver), cần chia lại
    scaling_factor = config.get('scaling_factor', 100)

    print(f"[*] Quy mô: {len(matrix)} node | {K} xe | Capacity: {Q} | "
          f"Scaling: {scaling_factor}")
    print(f"--- ĐANG GIẢI BẰNG MILP (Giới hạn: {timelimit}s) ---")

    start_time = time.time()
    status_str, obj_val, routes_info = solve_acvrp_milp(
        matrix, demands,
        num_vehicles=K,
        capacity=Q,
        timelimit=timelimit
    )
    duration = time.time() - start_time

    print("\n" + "=" * 50)
    print(f"TRẠNG THÁI SOLVER: {status_str}")

    if obj_val is None:
        print("[KẾT QUẢ] Solver không tìm được nghiệm khả thi.")
        print("=" * 50)
        exit(0)

    # ── Chuẩn hóa kết quả ──
    routes_dict = {}
    for idx, info in enumerate(routes_info):
        route = info['route']
        # Đảm bảo tuyến kết thúc bằng depot (đã xử lý trong _extract_routes,
        # thêm check ở đây để chắc chắn)
        if route[-1] != 0:
            route.append(0)
        routes_dict[idx] = route

    # FIX: Dùng scaling_factor từ config thay vì hard-code /100
    standardized_result = {
        "solver_name": "MILP",
        "total_distance_km": obj_val / scaling_factor,
        "execution_time": duration,
        "routes": routes_dict,
        "num_vehicles": len(routes_dict)
    }

    # ── In kết quả ──
    print(f"TỔNG KHOẢNG CÁCH: {standardized_result['total_distance_km']:.2f} km")
    print(f"Số xe sử dụng:    {standardized_result['num_vehicles']}")
    print(f"Thời gian chạy:   {duration:.2f} s")

    # Cảnh báo nếu có tuyến vi phạm capacity
    invalid_routes = [i for i, info in enumerate(routes_info) if not info['is_valid']]
    if invalid_routes:
        print(f"[CẢNH BÁO] {len(invalid_routes)} tuyến vi phạm capacity: {invalid_routes}")

    print("=" * 50)

    # ── Lưu kết quả ──
    output_dir = os.path.join(PROJECT_ROOT, "Results", "MILP")
    os.makedirs(output_dir, exist_ok=True)
    ResultHandler.save_to_txt(standardized_result, output_dir)
    ResultHandler.save_to_json(standardized_result, output_dir)

    # ── Trực quan hóa ──
    print("--- Đang khởi tạo bản đồ trực quan ---")
    try:
        if not os.path.exists(loc_path):
            print(f"[WARNING] Không tìm thấy file locations: {loc_path}")
        else:
            df_locations = pd.read_csv(loc_path)
            vis = Visualizer(
                df_locations,
                osrm_url="http://localhost:5001",
                use_osrm=True
            )
            map_path = os.path.join(output_dir, "route_map.html")
            vis.draw(standardized_result['routes'], map_path)
            print(f"[HOÀN TẤT] Bản đồ lưu tại: {map_path}")
    except Exception as e:
        print(f"[WARNING] Trực quan hóa thất bại: {e}")