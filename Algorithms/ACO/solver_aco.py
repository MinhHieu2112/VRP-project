import pandas as pd
import numpy as np
from Models.cvrp_base import CVRPGraph, Node
from Core.engine import BasicACO
import os
import sys
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer

print("Starting ACO solver...")


def load_data(locations_file, matrix_file):
    print(f"Loading data from:\n  {locations_file}\n  {matrix_file}")
    locations_df = pd.read_csv(locations_file)
    matrix_df    = pd.read_csv(matrix_file, header=None)

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
    routes       = {}
    vehicle_id   = 0
    current_route = [0]

    for node in best_path[1:]:
        current_route.append(node)
        if node == 0:
            if len(current_route) > 2:
                routes[vehicle_id] = current_route[:]
                vehicle_id += 1
            current_route = [0]

    if len(current_route) > 1:
        current_route.append(0)
        if len(current_route) > 2:
            routes[vehicle_id] = current_route

    return routes


def run_aco_solver():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Utils', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    locations_file   = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['locations_data'])
    matrix_file      = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['distance_matrix'])
    output_dir       = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['output_dir'])
    vehicle_capacity = config['global_constraints']['vehicle_capacity']

    # Đọc tham số ACO từ config (nếu có) hoặc dùng mặc định
    aco_cfg          = config.get('solvers', {}).get('aco', {})
    ants_num         = aco_cfg.get('ants_num', 20)
    max_iter         = aco_cfg.get('max_iter', 50000)
    no_improve_limit = aco_cfg.get('no_improve_limit', 1000)
    alpha            = aco_cfg.get('alpha', 1)
    beta             = aco_cfg.get('beta', 2)
    q0               = aco_cfg.get('q0', 0.9)

    print(f"Vehicle capacity: {vehicle_capacity}")
    print(f"ACO params: ants={ants_num}, max_iter={max_iter}, "
          f"no_improve_limit={no_improve_limit}")

    node_num, nodes, node_dist_mat = load_data(locations_file, matrix_file)

    print("Creating graph (with validation)...")
    graph = CVRPGraph(
        node_num, nodes, node_dist_mat, vehicle_capacity,
        rho=0.1,
        xi=0.01
    )

    print(f"\n--- ACO: dừng sau {no_improve_limit} vòng không cải thiện "
          f"(tối đa {max_iter} vòng) ---")

    start_time = time.time()
    aco = BasicACO(
        graph,
        ants_num=ants_num,
        max_iter=max_iter,
        alpha=alpha,
        beta=beta,
        q0=q0,
        no_improve_limit=no_improve_limit,
    )
    best_path, best_distance, best_vehicles = aco.run_basic_aco()
    execution_time = time.time() - start_time

    print(f"\nACO Done: dist={best_distance/1000:.2f}km, "
          f"vehicles={best_vehicles}, time={execution_time:.2f}s")

    routes_dict = parse_routes(best_path)

    served      = set()
    for route in routes_dict.values():
        for node in route:
            if node != 0:
                served.add(node)
    all_customers = set(range(1, node_num))
    missing = all_customers - served
    if missing:
        print(f"[WARN] {len(missing)} customer chưa được phục vụ: {sorted(missing)[:10]}")
    else:
        print(f"[OK] Tất cả {len(all_customers)} customer đã được phục vụ")

    standardized_result = {
        "solver_name":       "ACO",
        "total_distance_km": best_distance / 1000,
        "execution_time":    execution_time,
        "routes":            routes_dict,
        "num_vehicles":      best_vehicles
    }

    aco_output_dir = os.path.join(output_dir, 'ACO')
    os.makedirs(aco_output_dir, exist_ok=True)
    ResultHandler.save_to_txt(standardized_result, aco_output_dir)

    print("--- Visualizing ---")
    try:
        locations_df = pd.read_csv(locations_file)
        vis_config   = config.get('visualization', {})
        visualizer   = Visualizer(
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
    except ValueError as e:
        print(f"[INPUT ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)