import numpy as np
import time
import json
import os
import sys
import pandas as pd
from src.utils.loader import load_distance_matrix
from src.state import CvrpState
from src.solver import configure_alns
from alns.stop import MaxRuntime

# Import Utils chung từ project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer

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

    # === TẠO KẾT QUẢ CHUẨN ===
    actual_routes = [r for r in best_state.routes if len(r) > 2]
    final_distance = sum(best_state.route_cost(r) for r in actual_routes) / 100

    # Chuyển sang routes_dict chuẩn: {int: [0, ..., 0]}
    routes_dict = {}
    for idx, route in enumerate(actual_routes):
        # Đảm bảo route có depot đầu và cuối
        route_int = [int(n) for n in route]
        if route_int[0] != 0:
            route_int = [0] + route_int
        if route_int[-1] != 0:
            route_int = route_int + [0]
        routes_dict[idx] = route_int

    standardized_result = {
        "solver_name": "ALNS",
        "total_distance_km": final_distance,
        "execution_time": end_time - start_time,
        "routes": routes_dict,
        "num_vehicles": len(routes_dict)
    }

    # === LƯU KẾT QUẢ BẰNG RESULTHANDLER CHUNG ===
    output_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'Results', 'ALNS'
    )
    os.makedirs(output_dir, exist_ok=True)

    ResultHandler.save_to_txt(standardized_result, output_dir)
    ResultHandler.save_to_json(standardized_result, output_dir)

    # === IN KẾT QUẢ ===
    print(f"\nTổng quãng đường sau tối ưu: {final_distance:.2f} km")
    print(f"Số xe sử dụng: {len(routes_dict)}")
    print(f"Số khách hàng chưa gán: {len(best_state.unassigned)}")
    print(f"Thời gian tính toán: {end_time - start_time:.2f} giây")

    # === TRỰC QUAN HÓA BẰNG VISUALIZER CHUNG ===
    print("--- Đang khởi tạo bản đồ trực quan ---")
    try:
        vis_config = config.get('visualization', {})
        loc_path = config.get('locations_path', '../../Data/locations.csv')

        if os.path.exists(loc_path):
            df_locations = pd.read_csv(loc_path)

            vis = Visualizer(
                df_locations,
                osrm_url=vis_config.get('osrm_url', "http://localhost:5001"),
                use_osrm=vis_config.get('use_osrm', True)
            )

            # Dùng trực tiếp routes_dict chuẩn (không cần load_routes_from_txt)
            output_map_path = os.path.join(output_dir, vis_config.get('map_filename', "route_map.html"))
            vis.draw(standardized_result['routes'], output_map_path)
            print(f"[HOÀN TẤT] Bản đồ lưu tại: {output_map_path}")
        else:
            print(f"Thiếu file locations: {loc_path}")

    except Exception as e:
        print(f"Lỗi khi vẽ bản đồ: {e}")

if __name__ == "__main__":
    main()
