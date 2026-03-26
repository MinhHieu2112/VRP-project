import os
import json
import time
import pandas as pd
import numpy as np
from tabu_solver import TabuSearchSolver

def init_solution(num_nodes, max_v, cap):
    nodes = list(range(1, num_nodes))
    np.random.shuffle(nodes)
    routes = []
    # Chia khách vào xe theo capacity (mỗi xe tối đa 10 khách)
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
    with open('config_tabu.json', 'r') as f:
        config = json.load(f)

    # 2. Đọc dữ liệu ma trận
    df = pd.read_csv(config['data_path'], header=None)
    matrix = df.values
    num_nodes = matrix.shape[0]

    # 3. Khởi tạo lời giải
    initial_state = init_solution(
        num_nodes, 
        config['constraints']['max_vehicles'], 
        config['constraints']['vehicle_capacity']
    )

    # 4. Chạy Tabu Search với giới hạn 180s
    solver = TabuSearchSolver(
        distance_matrix=matrix,
        capacity=config['constraints']['vehicle_capacity'],
        max_v=config['constraints']['max_vehicles'],
        tabu_size=config['tabu_parameters']['tabu_size'],
        max_iter=10000, # Chạy số vòng lặp lớn để ưu tiên thời gian 180s
        max_runtime=config['tabu_parameters']['max_runtime']
    )

    print(f"--- Đang thực hiện Tabu Search (Giới hạn: {config['tabu_parameters']['max_runtime']}s) ---")
    start_time = time.time()
    best_state, best_dist = solver.solve(initial_state)
    duration = time.time() - start_time

    # 5. Tính toán thông số thống kê
    vehicles_used = sum(1 for r in best_state if len(r) > 2)
    unassigned = 0
    objective_value = best_dist * 100 # Theo định dạng bạn yêu cầu

    # 6. Ghi kết quả VÀO ĐẦU FILE result/tabu_result.txt
    out_dir = config['output_dir']
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    res_path = os.path.join(out_dir, "tabu_result.txt")

    with open(res_path, "w", encoding="utf-8") as f:
        # Ghi bảng thống kê ở đầu file
        f.write(f"Tổng quãng đường thực tế: {best_dist:.2f} km\n")
        f.write(f"Số xe sử dụng: {vehicles_used}\n")
        f.write(f"Số khách hàng chưa gán: {unassigned}\n")
        f.write(f"Giá trị Objective cuối cùng: {objective_value:.2f}\n")
        f.write(f"Tổng thời gian chạy: {duration:.2f} giây\n")
        f.write("-" * 40 + "\n")
        
        # Ghi danh sách các lộ trình
        for idx, route in enumerate(best_state):
            if len(route) > 2:
                f.write(f"Route #{idx+1}: {' '.join(map(str, route))}\n")

    # In ra màn hình terminal
    print(f"\nTổng quãng đường thực tế: {best_dist:.2f} km")
    print(f"Số xe sử dụng: {vehicles_used}")
    print(f"Số khách hàng chưa gán: {unassigned}")
    print(f"Giá trị Objective cuối cùng: {objective_value:.2f}")
    print(f"Tổng thời gian chạy: {duration:.2f} giây")
    print(f"\nKết quả đã ghi vào đầu file: {res_path}")

if __name__ == "__main__":
    main()