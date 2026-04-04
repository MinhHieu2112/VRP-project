import os
import sys
import json
import time
import pandas as pd
import numpy as np
from tabu_solver import TabuSearchSolver

# ===== XÁC ĐỊNH ĐƯỜNG DẪN =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

sys.path.append(PROJECT_ROOT)
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer


def load_demands(config, num_nodes):
    """
    Đọc demand của từng khách hàng từ config.

    Config hỗ trợ 2 dạng:
      - "demands": 1          → tất cả khách hàng có demand = 1 (đồng nhất)
      - "demands": [0,1,2,...] → demand riêng lẻ theo từng node

    FIX: Phiên bản cũ không đọc demand — mặc định coi demand = 1 ngầm định,
    dẫn đến kiểm tra capacity sai khi bài toán có demand không đồng nhất.

    Args:
        config:    Dict cấu hình đã đọc từ JSON
        num_nodes: Tổng số node (bao gồm depot)

    Returns:
        Dict {node_id: demand}, depot (node 0) luôn có demand = 0
    """
    demands = {0: 0}
    raw = config.get('demands', 1)

    if isinstance(raw, int):
        # Tất cả khách hàng có cùng demand
        for i in range(1, num_nodes):
            demands[i] = raw
    elif isinstance(raw, list):
        if len(raw) < num_nodes:
            raise ValueError(
                f"Danh sách demands ({len(raw)}) ngắn hơn số node ({num_nodes})."
            )
        for i in range(1, num_nodes):
            demands[i] = raw[i]
    else:
        raise TypeError("'demands' trong config phải là số nguyên hoặc list.")

    return demands


def init_solution(num_nodes, demands, max_v, capacity):
    """
    Tạo lời giải ban đầu bằng thuật toán GREEDY đơn giản:
    Lần lượt gán khách hàng vào xe hiện tại;
    nếu xe đầy (tổng demand + demand mới > capacity) thì chuyển sang xe tiếp theo.

    FIX: Phiên bản cũ dùng `cap` như số node tối đa thay vì tổng demand,
    gây ra vi phạm capacity khi demand > 1. Nay kiểm tra demand thực tế.

    Args:
        num_nodes: Tổng số node (bao gồm depot node 0)
        demands:   Dict {node_id: demand}
        max_v:     Số xe tối đa
        capacity:  Sức chứa tối đa mỗi xe

    Returns:
        List các route khả thi, mỗi route dạng [0, ..., 0]
    """
    # Xáo trộn ngẫu nhiên thứ tự khách hàng
    customers = list(range(1, num_nodes))
    np.random.shuffle(customers)

    routes = [[0, 0] for _ in range(max_v)]  # Khởi tạo max_v xe rỗng
    route_loads = [0] * max_v                # Tải hiện tại của từng xe

    for customer in customers:
        d = demands.get(customer, 1)
        assigned = False

        for v in range(max_v):
            if route_loads[v] + d <= capacity:
                # Chèn trước depot cuối cùng
                routes[v].insert(-1, customer)
                route_loads[v] += d
                assigned = True
                break

        if not assigned:
            # Không có xe nào chứa được → cảnh báo (nên tăng max_v hoặc capacity)
            print(f"[CẢNH BÁO] Không thể gán khách hàng {customer} "
                  f"(demand={d}) vào bất kỳ xe nào! "
                  f"Hãy tăng max_vehicles hoặc vehicle_capacity.")

    return routes


def main():
    """
    Hàm chính: đọc config → đọc dữ liệu → khởi tạo lời giải
    → chạy Tabu Search → lưu kết quả → vẽ bản đồ.
    """

    # ── 1. Đọc cấu hình ──
    config_path = os.path.join(CURRENT_DIR, 'config_tabu.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # ── 2. Đọc ma trận khoảng cách ──
    data_path = os.path.normpath(os.path.join(CURRENT_DIR, config['data_path']))
    if not os.path.exists(data_path):
        data_path = os.path.join(PROJECT_ROOT, "Data", "orsm_matrix.csv")

    df = pd.read_csv(data_path, header=None)
    matrix = df.values
    num_nodes = matrix.shape[0]
    print(f"[*] Đã đọc ma trận {num_nodes}x{num_nodes}")

    # ── 3. Đọc demand ──
    demands = load_demands(config['constraints'], num_nodes)

    # ── 4. Đọc locations để vẽ bản đồ ──
    loc_path = os.path.normpath(
        os.path.join(CURRENT_DIR, config.get('locations_path', '../../Data/locations.csv'))
    )
    if not os.path.exists(loc_path):
        loc_path = os.path.join(PROJECT_ROOT, "Data", "locations.csv")
    df_locations = pd.read_csv(loc_path)

    # ── 5. Khởi tạo lời giải ban đầu ──
    constraints = config['constraints']
    initial_state = init_solution(
        num_nodes,
        demands,
        constraints['max_vehicles'],
        constraints['vehicle_capacity']
    )

    # ── 6. Chạy Tabu Search ──
    tabu_params = config['tabu_parameters']
    solver = TabuSearchSolver(
        distance_matrix=matrix,
        demands=demands,                            # ← THÊM: truyền demand thực tế
        capacity=constraints['vehicle_capacity'],
        max_v=constraints['max_vehicles'],
        tabu_size=tabu_params['tabu_size'],
        max_iter=tabu_params.get('max_iterations', 10000),
        max_runtime=tabu_params['max_runtime']
    )

    print(f"\n--- Đang thực hiện Tabu Search "
          f"(Giới hạn: {tabu_params['max_runtime']}s) ---")
    start_time = time.time()
    best_state, best_dist = solver.solve(initial_state)
    duration = time.time() - start_time

    # ── 7. Chuẩn hóa kết quả ──
    routes_dict = {}
    idx = 0
    for route in best_state:
        if len(route) > 2:          # Bỏ qua xe rỗng [0, 0]
            routes_dict[idx] = route
            idx += 1

    standardized_result = {
        "solver_name": "Tabu Search",
        "total_distance_km": best_dist,
        "execution_time": duration,
        "routes": routes_dict,
        "num_vehicles": len(routes_dict)
    }

    # ── 8. In kết quả ──
    print(f"\n{'='*50}")
    print(f"Tổng quãng đường: {standardized_result['total_distance_km']:.2f} km")
    print(f"Số xe sử dụng:    {standardized_result['num_vehicles']}")
    print(f"Thời gian chạy:   {duration:.2f} giây")
    print('='*50)

    # ── 9. Lưu kết quả ──
    output_dir = os.path.join(PROJECT_ROOT, "Results", "Tabu")
    os.makedirs(output_dir, exist_ok=True)
    ResultHandler.save_to_txt(standardized_result, output_dir)
    ResultHandler.save_to_json(standardized_result, output_dir)

    # ── 10. Trực quan hóa ──
    print("--- Đang khởi tạo bản đồ trực quan ---")
    try:
        vis = Visualizer(df_locations, osrm_url="http://localhost:5001", use_osrm=True)
        map_path = os.path.join(output_dir, "route_map.html")
        vis.draw(standardized_result['routes'], map_path)
        print(f"[HOÀN TẤT] Bản đồ lưu tại: {map_path}")
    except Exception as e:
        print(f"[WARNING] Trực quan hóa thất bại: {e}")


if __name__ == "__main__":
    main()