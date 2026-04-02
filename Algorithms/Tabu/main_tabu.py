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

# Import Utils chung từ project root
sys.path.append(PROJECT_ROOT)
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer

def init_solution(num_nodes, max_v, cap):
    nodes = list(range(1, num_nodes))
    np.random.shuffle(nodes)
    routes = []
    for i in range(0, len(nodes), cap):
        if len(routes) < max_v:
            r = [0] + nodes[i : i + cap] + [0]
            routes.append(r)
        else:
            routes[i % max_v].insert(-1, nodes[i])

    while len(routes) < max_v:
        routes.append([0, 0])
    return routes

def main():
    # 1. Đọc cấu hình
    config_path = os.path.join(CURRENT_DIR, 'config_tabu.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 2. Đọc dữ liệu ma trận
    data_path = os.path.normpath(os.path.join(CURRENT_DIR, config['data_path']))
    if not os.path.exists(data_path):
        data_path = os.path.join(PROJECT_ROOT, "Data", "orsm_matrix.csv")
    df = pd.read_csv(data_path, header=None)
    matrix = df.values
    num_nodes = matrix.shape[0]

    # Đọc locations cho visualization
    loc_path = os.path.normpath(os.path.join(CURRENT_DIR, config.get('locations_path', '../../Data/locations.csv')))
    if not os.path.exists(loc_path):
        loc_path = os.path.join(PROJECT_ROOT, "Data", "locations.csv")
    df_locations = pd.read_csv(loc_path)

    # 3. Khởi tạo lời giải
    initial_state = init_solution(
        num_nodes,
        config['constraints']['max_vehicles'],
        config['constraints']['vehicle_capacity']
    )

    # 4. Chạy Tabu Search
    solver = TabuSearchSolver(
        distance_matrix=matrix,
        capacity=config['constraints']['vehicle_capacity'],
        max_v=config['constraints']['max_vehicles'],
        tabu_size=config['tabu_parameters']['tabu_size'],
        max_iter=10000,
        max_runtime=config['tabu_parameters']['max_runtime']
    )

    print(f"--- Đang thực hiện Tabu Search (Giới hạn: {config['tabu_parameters']['max_runtime']}s) ---")
    start_time = time.time()
    best_state, best_dist = solver.solve(initial_state)
    duration = time.time() - start_time

    # === TẠO KẾT QUẢ CHUẨN ===
    routes_dict = {}
    idx = 0
    for route in best_state:
        if len(route) > 2:
            routes_dict[idx] = route
            idx += 1

    standardized_result = {
        "solver_name": "Tabu Search",
        "total_distance_km": best_dist,
        "execution_time": duration,
        "routes": routes_dict,
        "num_vehicles": len(routes_dict)
    }

    # === IN KẾT QUẢ ===
    print(f"\nTổng quãng đường thực tế: {standardized_result['total_distance_km']:.2f} km")
    print(f"Số xe sử dụng: {standardized_result['num_vehicles']}")
    print(f"Tổng thời gian chạy: {duration:.2f} giây")

    # === LƯU KẾT QUẢ BẰNG RESULTHANDLER CHUNG ===
    output_dir = os.path.join(PROJECT_ROOT, "Results", "Tabu")
    os.makedirs(output_dir, exist_ok=True)

    ResultHandler.save_to_txt(standardized_result, output_dir)
    ResultHandler.save_to_json(standardized_result, output_dir)

    # === TRỰC QUAN HÓA BẰNG VISUALIZER CHUNG ===
    print("--- Đang khởi tạo bản đồ trực quan ---")
    try:
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

if __name__ == "__main__":
    main()
