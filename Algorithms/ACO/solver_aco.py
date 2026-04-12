"""
Algorithms/ACO/solver_aco.py
FIXES:
  [FIX-1] DataLoader.load_data() trả dict, không phải tuple → unpack đúng.
  [FIX-2] load_config dùng realpath(__file__) thay 'config.json' relative.
  [FIX-3] output_dir dùng _PROJECT_ROOT, không double-prefix config path.
  [FIX-5] parse_routes xử lý route cuối không kết thúc bằng 0.
  [FIX-SEED] seed_weight đọc từ config (1.3 thay 2.0) để giảm lock greedy.
  [REFACTOR] Tách các logic trong run_aco_solver() thành các hàm con.
"""

import os, sys, json, time
import numpy as np

_THIS_DIR     = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, '..', '..'))
sys.path.append(_PROJECT_ROOT)

from Models.cvrp_base import CVRPGraph, Node
from Core.engine import BasicACO
from Algorithms.Init_strategies.Init_strategies import init_solution
from Utils.Data_loader import DataLoader
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer

KM_SCALE = 100


def load_config() -> dict:
    config_path = os.path.join(_THIS_DIR, 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def load_data_from_loader(config: dict) -> tuple:
    print("Initializing DataLoader...")
    data             = DataLoader(config).load_data()     # [FIX-1] dict, not tuple
    matrix_int       = data['distance_matrix']
    vehicle_capacity = data['vehicle_capacity']
    df_locs          = data['df_locations']
    demands          = data['demands']
    nodes = [
        Node(int(row['id']), float(row['lat']), float(row['lon']), float(demands[idx]))
        for idx, row in df_locs.iterrows()
    ]
    return len(nodes), nodes, matrix_int.astype(np.float64), vehicle_capacity, df_locs


def parse_routes(best_path: list) -> dict:
    routes, current_route, vehicle_id = {}, [0], 0
    for node in best_path[1:]:
        current_route.append(node)
        if node == 0:
            if len(current_route) > 2:
                routes[vehicle_id] = current_route[:]
                vehicle_id += 1
            current_route = [0]
    if len(current_route) > 1:             # [FIX-5] route cuối
        current_route.append(0)
        if len(current_route) > 2:
            routes[vehicle_id] = current_route[:]
    return routes


def setup_aco_engine(graph: CVRPGraph, aco_cfg: dict) -> BasicACO:
    return BasicACO(
        graph,
        ants_num        = aco_cfg.get('ants_num',          20),
        max_iter        = aco_cfg.get('max_iter',         2000),
        alpha           = aco_cfg.get('alpha',             1.0),
        beta            = aco_cfg.get('beta',              2.0),
        q0              = aco_cfg.get('q0',                0.9),
        no_improve_limit= aco_cfg.get('no_improve_limit',  500),
    )


def save_and_visualize(routes, best_dist, best_v, elapsed, config, df_locs):
    #actual_vehicles = len(routes) if routes else 0
    result = {
        "solver_name":       "ACO",
        "total_distance_km": best_dist / KM_SCALE,
        "execution_time":    elapsed,
        "routes":            routes,
        "num_vehicles":      best_v,
    }
    output_dir = os.path.join(_PROJECT_ROOT, 'Results', 'ACO')  # [FIX-3]
    os.makedirs(output_dir, exist_ok=True)
    ResultHandler.save_to_txt(result, output_dir)

    vis_cfg  = config.get('visualization', {})
    map_path = os.path.join(output_dir, vis_cfg.get('map_filename', 'route_map.html'))
    try:
        vis = Visualizer(df_locs, osrm_url=vis_cfg.get('osrm_url', 'http://localhost:5001'),
                         use_osrm=vis_cfg.get('use_osrm', True))
        vis.draw(routes, map_path)
        print(f"[Visualizer] Bản đồ: {map_path}")
    except Exception as exc:
        print(f"[Visualizer] Thất bại: {exc}")

    print(f"\n[ACO DONE] {result['total_distance_km']:.2f} km | "
          f"{result['num_vehicles']} xe | {elapsed:.2f}s")


def _initialize_seed(dist_mat, num_nodes, capacity, aco_cfg):
    """Xử lý khởi tạo nghiệm ban đầu"""
    seed_strategy = aco_cfg.get('seed_strategy', '')
    seed_weight   = aco_cfg.get('seed_weight', 1.3)
    print(f"Building seed solution ({seed_strategy})...")
    
    seed_sol  = init_solution(seed_strategy, dist_mat, num_nodes, capacity,
                               default_demand=1.0, max_vehicles=200, validate=True)
    seed_cost = sum(dist_mat[r[i], r[i+1]] for r in seed_sol for i in range(len(r)-1))
    
    print(f"[ACO Init] Đã nạp mốc {seed_strategy.capitalize()}: "
          f"Cost={seed_cost:.0f} units ({seed_cost/KM_SCALE:.2f} km), "
          f"Xe={len(seed_sol)}, seed_weight={seed_weight}")
          
    return seed_sol, seed_cost, seed_weight

def _execute_aco(graph, aco_cfg, seed_sol, seed_cost):
    """Thiết lập và chạy thuật toán ACO"""
    aco = setup_aco_engine(graph, aco_cfg)
    
    # --- ĐOẠN SỬA LỖI (FIX) ---
    # Chuyển đổi seed_sol (list of lists) thành flat list
    # Ví dụ: [[0, 1, 2, 0], [0, 3, 4, 0]] -> [0, 1, 2, 0, 3, 4, 0]
    flat_seed_path = []
    for route in seed_sol:
        if not flat_seed_path:
            flat_seed_path.extend(route) # Add route đầu tiên
        else:
            flat_seed_path.extend(route[1:]) # Bỏ qua số 0 ở đầu để tránh trùng lặp 0, 0
    # --------------------------

    aco.best_path_distance = float(seed_cost) 
    # Gán flat list đã được làm phẳng thay vì list of lists ban đầu
    aco.best_path = flat_seed_path 
    
    start = time.time()
    best_path, best_dist, best_v = aco.run_basic_aco()
    elapsed = time.time() - start
    
    return best_path, best_dist, best_v, elapsed


def run_aco_solver():
    config  = load_config()
    aco_cfg = config.get('solvers', {}).get('aco', {})

    num_nodes, nodes, dist_mat, capacity, df_locs = load_data_from_loader(config)
    graph = CVRPGraph(num_nodes, nodes, dist_mat, capacity, rho=0.1, xi=0.01)

    seed_sol, seed_cost, seed_weight = _initialize_seed(dist_mat, num_nodes, capacity, aco_cfg)
    graph.seed_pheromone(seed_sol, seed_weight=seed_weight)

    best_path, best_dist, best_v, elapsed = _execute_aco(graph, aco_cfg, seed_sol, seed_cost)

    routes = parse_routes(best_path)
    save_and_visualize(routes, best_dist, best_v, elapsed, config, df_locs)


if __name__ == '__main__':
    run_aco_solver()