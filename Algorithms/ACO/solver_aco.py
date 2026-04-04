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

print("Starting ACO solver (Fixed Version)...")


def load_data(locations_file, matrix_file):
    print(f"Loading data from:\n  {locations_file}\n  {matrix_file}")
    locations_df = pd.read_csv(locations_file)
    matrix_df = pd.read_csv(matrix_file, header=None)

    nodes = []
    for idx, row in locations_df.iterrows():
        node = Node(
            id=int(row['id']),
            x=float(row['lat']),
            y=float(row['lon']),
            demand=1
        )
        nodes.append(node)

    node_dist_mat = matrix_df.values.astype(np.float64)
    print(f"Loaded {len(nodes)} nodes, matrix shape: {node_dist_mat.shape}")
    return len(nodes), nodes, node_dist_mat


def parse_routes(best_path: list) -> dict:
    """
    Parse flat path (vd [0,1,2,0,3,4,0]) thành dict routes.
    [FIX S4] Bỏ qua route rỗng (depot→depot).
    """
    routes = {}
    vehicle_id = 0
    current_route = [0]

    for node in best_path[1:]:
        current_route.append(node)
        if node == 0:
            # Chỉ lưu nếu route có ít nhất 1 customer
            if len(current_route) > 2:
                routes[vehicle_id] = current_route[:]
                vehicle_id += 1
            current_route = [0]

    # Route cuối chưa kết thúc bằng 0
    if len(current_route) > 1:
        current_route.append(0)
        if len(current_route) > 2:
            routes[vehicle_id] = current_route

    return routes


def run_aco_solver():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Utils', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    locations_file = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['locations_data'])
    matrix_file = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['distance_matrix'])
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['output_dir'])
    vehicle_capacity = config['global_constraints']['vehicle_capacity']
    scaling_factor = config['common_model_parameters']['scaling_factor']

    print(f"Vehicle capacity: {vehicle_capacity}")

    node_num, nodes, node_dist_mat = load_data(locations_file, matrix_file)

    print("Creating graph...")
    graph = CVRPGraph(
        node_num, nodes, node_dist_mat, vehicle_capacity,
        rho=0.1,   # Global evaporation
        xi=0.01    # [FIX C4] Local update rate (ACS gốc ξ = 0.1, thử nhỏ hơn cho ổn định)
    )

    print("Running ACO...")
    start_time = time.time()
    aco = BasicACO(
        graph,
        ants_num=20,
        max_iter=100,
        alpha=1,        # [FIX C1] Alpha rõ ràng
        beta=2,
        q0=0.9,         # [FIX M3] Exploitation-first
        no_improve_limit=50
    )
    best_path, best_distance, best_vehicles = aco.run_basic_aco()
    execution_time = time.time() - start_time

    print(f"\nACO Done: dist={best_distance:.2f}, vehicles={best_vehicles}, time={execution_time:.2f}s")

    routes_dict = parse_routes(best_path)

    standardized_result = {
        "solver_name": "ACO_Fixed",
        "total_distance_km": best_distance / scaling_factor,
        "execution_time": execution_time,
        "routes": routes_dict,
        "num_vehicles": best_vehicles
    }

    aco_output_dir = os.path.join(output_dir, 'ACO')
    os.makedirs(aco_output_dir, exist_ok=True)
    ResultHandler.save_to_txt(standardized_result, aco_output_dir)
    ResultHandler.save_to_json(standardized_result, aco_output_dir)

    print("--- Visualizing ---")
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
        print(f"Map saved: {map_path}")
    except Exception as e:
        print(f"[WARNING] Visualization failed: {e}")

    print(f"\nFinal: {standardized_result['total_distance_km']:.2f} km, "
          f"{best_vehicles} vehicles, {execution_time:.2f}s")


if __name__ == '__main__':
    try:
        run_aco_solver()
        print("ACO solver completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()