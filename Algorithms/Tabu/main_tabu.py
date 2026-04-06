import os
import sys
import json
import time
import pandas as pd
import numpy as np
from tabu_solver import TabuSearchSolver

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

sys.path.append(PROJECT_ROOT)
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer


def load_demands(config, num_nodes):
    demands = {0: 0}
    raw = config.get('demands', 1)

    if isinstance(raw, int):
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
    customers    = list(range(1, num_nodes))
    np.random.shuffle(customers)

    routes       = [[0, 0] for _ in range(max_v)]
    route_loads  = [0] * max_v

    for customer in customers:
        d        = demands.get(customer, 1)
        assigned = False

        for v in range(max_v):
            if route_loads[v] + d <= capacity:
                routes[v].insert(-1, customer)
                route_loads[v] += d
                assigned = True
                break

        if not assigned:
            print(f"[CẢNH BÁO] Không thể gán khách hàng {customer} "
                  f"(demand={d}) vào bất kỳ xe nào!")

    return routes


def main():
    config_path = os.path.join(CURRENT_DIR, 'config_tabu.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    data_path = os.path.normpath(os.path.join(CURRENT_DIR, config['data_path']))
    if not os.path.exists(data_path):
        data_path = os.path.join(PROJECT_ROOT, "Data", "orsm_matrix.csv")

    df         = pd.read_csv(data_path, header=None)
    matrix     = df.values
    num_nodes  = matrix.shape[0]
    print(f"[*] Đã đọc ma trận {num_nodes}×{num_nodes}")

    demands = load_demands(config['constraints'], num_nodes)

    loc_path = os.path.normpath(
        os.path.join(CURRENT_DIR, config.get('locations_path', '../../Data/locations.csv'))
    )
    if not os.path.exists(loc_path):
        loc_path = os.path.join(PROJECT_ROOT, "Data", "locations.csv")
    df_locations = pd.read_csv(loc_path)

    constraints  = config['constraints']
    tabu_params  = config['tabu_parameters']
    initial_state = init_solution(
        num_nodes,
        demands,
        constraints['max_vehicles'],
        constraints['vehicle_capacity']
    )

    max_no_improve = tabu_params.get('max_no_improve', 1000)
    max_iterations = tabu_params.get('max_iterations', 50000)
    print(f"\n--- Tabu Search | dừng sau {max_no_improve} vòng không cải thiện "
          f"(tối đa {max_iterations} vòng) ---")

    solver = TabuSearchSolver(
        distance_matrix=matrix,
        demands=demands,
        capacity=constraints['vehicle_capacity'],
        max_v=constraints['max_vehicles'],
        tabu_size=tabu_params['tabu_size'],
        max_iter=max_iterations,
        max_no_improve=max_no_improve,
    )

    start_time         = time.time()
    best_state, best_dist = solver.solve(initial_state)
    duration           = time.time() - start_time

    scaling_factor = config.get('common_model_parameters', {}).get('scaling_factor', 1000)

    routes_dict = {}
    idx = 0
    for route in best_state:
        if len(route) > 2:
            routes_dict[idx] = route
            idx += 1

    standardized_result = {
        "solver_name":        "Tabu Search",
        "total_distance_km":  best_dist / scaling_factor,
        "execution_time":     duration,
        "routes":             routes_dict,
        "num_vehicles":       len(routes_dict)
    }

    print(f"\n{'='*50}")
    print(f"Tổng quãng đường: {standardized_result['total_distance_km']:.2f} km")
    print(f"Số xe sử dụng:    {standardized_result['num_vehicles']}")
    print(f"Thời gian chạy:   {duration:.2f} giây")
    print('='*50)

    output_dir = os.path.join(PROJECT_ROOT, "Results", "Tabu")
    os.makedirs(output_dir, exist_ok=True)
    ResultHandler.save_to_txt(standardized_result, output_dir)

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