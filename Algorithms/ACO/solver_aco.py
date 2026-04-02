import pandas as pd
import numpy as np
from cvrp_base import CVRPGraph, Node
from basic_aco import BasicACO
import os
import sys
import json
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer

print("Starting ACO solver...")

def load_data(locations_file, matrix_file):
    print(f"Loading data from {locations_file} and {matrix_file}")
    locations_df = pd.read_csv(locations_file)
    matrix_df = pd.read_csv(matrix_file, header=None)

    nodes = []
    for idx, row in locations_df.iterrows():
        node = Node(id=int(row['id']), x=float(row['lat']), y=float(row['lon']), demand=1)
        nodes.append(node)

    node_dist_mat = matrix_df.values
    print(f"Loaded {len(nodes)} nodes and distance matrix of shape {node_dist_mat.shape}")
    return len(nodes), nodes, node_dist_mat


def run_aco_solver():
    print("Loading config...")
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Utils', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    locations_file = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['locations_data'])
    matrix_file = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['distance_matrix'])
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['output_dir'])
    vehicle_capacity = config['global_constraints']['vehicle_capacity']
    scaling_factor = config['common_model_parameters']['scaling_factor']

    print(f"Config loaded. Vehicle capacity: {vehicle_capacity}")

    # Load data
    node_num, nodes, node_dist_mat = load_data(locations_file, matrix_file)

    print("Creating graph...")
    # Create graph
    graph = CVRPGraph(node_num, nodes, node_dist_mat, vehicle_capacity)

    print("Running ACO...")
    # Run ACO với đo thời gian
    start_time = time.time()
    aco = BasicACO(graph, ants_num=20, max_iter=100, beta=2, q0=0.1, whether_or_not_to_show_figure=False)
    best_path, best_distance, best_vehicles = aco.run_basic_aco()
    execution_time = time.time() - start_time

    print(f"ACO completed. Distance: {best_distance}, Vehicles: {best_vehicles}")

    # Parse best_path into routes dictionary
    routes_dict = {}
    current_route = [0]
    vehicle_id = 0

    for i in range(1, len(best_path)):
        node = best_path[i]
        current_route.append(node)
        if node == 0:
            routes_dict[vehicle_id] = current_route.copy()
            vehicle_id += 1
            current_route = [0]

    # Handle the last route if it doesn't end with 0
    if len(current_route) > 1:
        current_route.append(0)
        routes_dict[vehicle_id] = current_route

    # === TẠO KẾT QUẢ CHUẨN ===
    standardized_result = {
        "solver_name": "ACO",
        "total_distance_km": best_distance / scaling_factor,
        "execution_time": execution_time,
        "routes": routes_dict,
        "num_vehicles": best_vehicles
    }

    # === LƯU KẾT QUẢ BẰNG RESULTHANDLER CHUNG ===
    aco_output_dir = os.path.join(output_dir, 'ACO')
    os.makedirs(aco_output_dir, exist_ok=True)
    ResultHandler.save_to_txt(standardized_result, aco_output_dir)
    ResultHandler.save_to_json(standardized_result, aco_output_dir)

    # === TRỰC QUAN HÓA BẰNG VISUALIZER CHUNG ===
    print("--- Đang khởi tạo bản đồ trực quan ---")
    try:
        locations_df = pd.read_csv(locations_file)
        vis_config = config.get('visualization', {})
        visualizer = Visualizer(
            locations_df,
            osrm_url=vis_config.get('osrm_url', "http://localhost:5001"),
            use_osrm=vis_config.get('use_osrm', True)
        )
        map_path = os.path.join(aco_output_dir, vis_config.get('map_filename', 'route_map.html'))
        visualizer.draw(standardized_result['routes'], map_path)
        print(f"[HOÀN TẤT] Bản đồ lưu tại: {map_path}")
    except Exception as e:
        print(f"[WARNING] Trực quan hóa thất bại: {e}")

    print(f"\nACO Result: Distance {standardized_result['total_distance_km']:.2f} km, Vehicles {best_vehicles}, Time {execution_time:.2f}s")


if __name__ == '__main__':
    try:
        run_aco_solver()
        print("ACO solver completed successfully!")
    except Exception as e:
        print(f"Error running ACO solver: {e}")
        import traceback
        traceback.print_exc()
