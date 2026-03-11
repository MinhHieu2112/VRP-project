import numpy as np
import time
import json
import os
import pandas as pd
from src.utils.loader import load_distance_matrix
from src.state import CvrpState
from src.solver import configure_alns
from alns.stop import MaxRuntime
from src.utils.visualizer import Visualizer

def load_config(path='config.json'):
    with open(path, 'r') as f:
        return json.load(f)

def generate_smart_initial_solution(num_nodes, max_v, capacity):
    clients = list(range(1, num_nodes))
    routes = []
    chunk_size = int(np.ceil(len(clients) / max_v))
    nodes_per_vehicle = min(chunk_size, capacity)
    
    for i in range(0, len(clients), nodes_per_vehicle):
        route = [0] + clients[i : i + nodes_per_vehicle] + [0]
        routes.append(route)
        if len(routes) == max_v:
            remaining = clients[i + nodes_per_vehicle:]
            for idx, node in enumerate(remaining):
                routes[idx % max_v].insert(-1, node)
            break
    return routes

def main():
    config = load_config()
    matrix = load_distance_matrix(config['data_path'])
    if matrix is None: return
    num_nodes = matrix.shape[0]

    demands = np.ones(num_nodes) * config['constraints']['default_demand']
    demands[0] = 0 
    
    initial_routes = generate_smart_initial_solution(
        num_nodes, config['constraints']['max_vehicles'], config['constraints']['vehicle_capacity']
    )
    
    initial_state = CvrpState(initial_routes, [], matrix, 
                              capacity=config['constraints']['vehicle_capacity'], 
                              demands=demands, config=config)

    print(f"Quãng đường ban đầu: {initial_state.objective() / 100:.2f} km")

    alns, accept, select, log_func = configure_alns(initial_state, config)
    alns.on_best = log_func
    stop_criterion = MaxRuntime(config['alns_parameters']['max_runtime'])
    
    print(f"--- Đang thực hiện tối ưu hóa ALNS ({config['alns_parameters']['max_runtime']} giây) ---")
    start_time = time.time()
    result = alns.iterate(initial_state, op_select=select, accept=accept, stop=stop_criterion)
    
    best_state = result.best_state

    # BƯỚC QUAN TRỌNG: Local Search để tối ưu sâu
    print("--- Đang áp dụng 2-opt Local Search để làm đẹp lộ trình ---")
    best_state.apply_2opt()
    
    end_time = time.time()
    os.makedirs(config['output_dir'], exist_ok=True)

    actual_routes = [r for r in best_state.routes if len(r) > 2]
    final_distance = sum(best_state.route_cost(r) for r in actual_routes) / 100

    print(f"Tổng quãng đường sau tối ưu: {final_distance:.2f} km")
    print(f"Số xe sử dụng: {len(actual_routes)}")
    print(f"Số khách hàng chưa gán: {len(best_state.unassigned)}")
    print(f"Thời gian tính toán: {end_time - start_time:.2f} giây")

    # Lưu kết quả
    with open(f"{config['output_dir']}/alns_result.txt", "w", encoding="utf-8") as f:
        f.write(f"Tổng quãng đường thực tế: {final_distance:.2f} km\n")
        f.write(f"Số xe sử dụng: {len(actual_routes)}\n")
        f.write(f"Số khách hàng chưa gán: {len(best_state.unassigned)}\n")
        f.write(f"Giá trị Objective cuối cùng: {best_state.objective():.2f}\n\n")
        f.write("Lộ trình chi tiết:\n")
        for idx, route in enumerate(actual_routes, 1):
            nodes_only = [str(int(node)) for node in route if node != 0]
            f.write(f"Route #{idx}: {' '.join(nodes_only)}\n")

    # 6. Vẽ bản đồ trực quan
    print("--- Đang khởi tạo bản đồ trực quan từ file kết quả ---")
    try:
        vis_config = config.get('visualization', {})
        loc_path = config.get('locations_path', '../Data/locations.csv')
        result_txt_path = f"{config['output_dir']}/alns_result.txt" # Đường dẫn file txt
        
        if os.path.exists(loc_path) and os.path.exists(result_txt_path):
            import pandas as pd
            df_locations = pd.read_csv(loc_path)
            
            # Khởi tạo Visualizer
            vis = Visualizer(df_locations, osrm_url=vis_config.get('osrm_url', "http://localhost:5001"))
            
            # BƯỚC MỚI: Đọc route từ file TXT thay vì dùng biến trực tiếp
            routes_from_file = vis.load_routes_from_txt(result_txt_path)
            
            if routes_from_file:
                output_map_path = os.path.join(config['output_dir'], vis_config.get('map_filename', "route_map.html"))
                vis.draw(routes_from_file, output_map_path)
                print(f"Thành công! Bản đồ được dựng từ file {result_txt_path}")
            else:
                print("Không thể parse dữ liệu từ file txt.")
        else:
            print(f"Thiếu file: {loc_path} hoặc {result_txt_path}")

    except Exception as e:
        print(f"Lỗi khi vẽ bản đồ: {e}")

if __name__ == "__main__":
    main()