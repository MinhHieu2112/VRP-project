# File điều phối chính chạy thuật toán ACO (Ant Colony Optimization) giải bài toán VRP.
import os, sys, json, time
import numpy as np

_THIS_DIR     = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, '..', '..'))
sys.path.append(_PROJECT_ROOT)

from Models.cvrp_base import CVRPGraph, Node
from Core.engine import BasicACO
from Algorithms.Init_strategies.Init_strategies import init_solution
from Utils.Pipeline import DataLoader, ResultHandler, Visualizer

KM_SCALE = 100

def load_config() -> dict:
    # Đọc thông tin cấu hình thuật toán từ tệp config.json.
    config_path = os.path.join(_THIS_DIR, 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def load_data_from_loader(config: dict) -> tuple:
    # Nạp dữ liệu bài toán và đồng bộ số lượng node với ma trận khoảng cách.
    print("Initializing DataLoader...")
    data             = DataLoader(config).load_data()
    matrix_int       = data['distance_matrix']
    vehicle_capacity = data['vehicle_capacity']
    df_locs          = data['df_locations']
    demands          = data['demands']

    num_nodes_from_matrix = matrix_int.shape[0]
    df_locs = df_locs.head(num_nodes_from_matrix)
    demands = demands[:num_nodes_from_matrix]

    nodes = [
        Node(int(row['id']), float(row['lat']), float(row['lon']), float(demands[idx]))
        for idx, row in df_locs.iterrows()
    ]
    
    print(f"[ACO] Đã đồng bộ: {len(nodes)} nodes khớp với ma trận {matrix_int.shape}")
    
    return len(nodes), nodes, matrix_int.astype(np.float64), vehicle_capacity, df_locs


def parse_routes(best_path: list) -> dict:
    # Chuyển đổi đường đi phẳng của kiến thành dict các tuyến đường xe.
    routes, current_route, vehicle_id = {}, [0], 0
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
            routes[vehicle_id] = current_route[:]
    return routes


def setup_aco_engine(graph: CVRPGraph, aco_cfg: dict) -> BasicACO:
    # Khởi tạo đối tượng ACO với các tham số từ cấu hình.
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
    # Lưu kết quả lời giải và tạo bản đồ trực quan hóa các tuyến đường.
    result = {
        "solver_name":       "ACO",
        "total_distance_km": best_dist / KM_SCALE,
        "execution_time":    elapsed,
        "routes":            routes,
        "num_vehicles":      best_v,
    }
    output_dir = os.path.join(_PROJECT_ROOT, 'Results', 'ACO')
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
    # Khởi tạo nghiệm mốc ban đầu để định hướng pheromone cho kiến.
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
    # Thiết lập và chạy vòng tối ưu hóa chính của thuật toán ACO.
    aco = setup_aco_engine(graph, aco_cfg)
    
    flat_seed_path = []
    for route in seed_sol:
        if not flat_seed_path:
            flat_seed_path.extend(route)
        else:
            flat_seed_path.extend(route[1:])

    aco.best_path_distance = float(seed_cost) 
    aco.best_path = flat_seed_path 
    
    start = time.time()
    best_path, best_dist, best_v = aco.run_basic_aco()
    elapsed = time.time() - start
    
    return best_path, best_dist, best_v, elapsed


def run_aco_solver():
    # Điều phối toàn bộ quy trình ACO từ nạp cấu hình đến lưu kết quả.
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