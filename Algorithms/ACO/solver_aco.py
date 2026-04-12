import os
import sys
import json
import time
import pandas as pd
import numpy as np

# Import các thành phần hệ thống
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Models.cvrp_base import CVRPGraph, Node
from Core.engine import BasicACO
from Algorithms.Init_strategies.Init_strategies import init_solution
from Utils.Data_loader import DataLoader
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer

# ==========================================
# 1. HÀM CẤU HÌNH & DỮ LIỆU
# ==========================================
def load_config():
    # config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Utils', 'config.json')
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        return json.load(f)

def load_data_from_loader(config):
    """Sử dụng DataLoader để đồng nhất dữ liệu."""
    print("Initializing DataLoader...")
    loader = DataLoader(config)
    num_vehicles, vehicle_capacity, matrix_int, df_locs, demands = loader.load_data()
    
    # Chuyển thành list Node cho mô hình
    nodes = [Node(int(row['id']), float(row['lat']), float(row['lon']), demands[idx]) 
             for idx, row in df_locs.iterrows()]
    
    return len(nodes), nodes, matrix_int.astype(np.float64), vehicle_capacity, df_locs

# ==========================================
# 2. HÀM XỬ LÝ LỘ TRÌNH
# ==========================================
def parse_routes(best_path: list) -> dict:
    routes, current_route, vehicle_id = {}, [0], 0
    for node in best_path[1:]:
        current_route.append(node)
        if node == 0:
            if len(current_route) > 2:
                routes[vehicle_id] = current_route[:]
                vehicle_id += 1
            current_route = [0]
    return routes

# ==========================================
# 3. HÀM THỰC THI THUẬT TOÁN
# ==========================================
def setup_aco_engine(graph, aco_cfg):
    return BasicACO(
        graph,
        ants_num=aco_cfg.get('ants_num', 20),
        max_iter=aco_cfg.get('max_iter', 50000),
        alpha=aco_cfg.get('alpha', 1),
        beta=aco_cfg.get('beta', 2),
        q0=aco_cfg.get('q0', 0.9),
        no_improve_limit=aco_cfg.get('no_improve_limit', 100)
    )

def run_aco_solver():
    config = load_config()
    num_nodes, nodes, dist_mat, capacity, df_locs = load_data_from_loader(config)
    
    # Khởi tạo Graph
    graph = CVRPGraph(num_nodes, nodes, dist_mat, capacity, rho=0.1, xi=0.01)
    
    # Khởi tạo Pheromone từ nghiệm Greedy (Seed)
    print("Seeding pheromone...")
    seed_sol = init_solution("greedy", dist_mat, num_nodes, capacity, 1.0, 200, True)
    graph.seed_pheromone(seed_sol, seed_weight=2.0)

    # Chạy Engine
    aco = setup_aco_engine(graph, config.get('solvers', {}).get('aco', {}))
    start_time = time.time()
    best_path, best_dist, best_v = aco.run_basic_aco()
    duration = time.time() - start_time

    # Xử lý kết quả
    routes = parse_routes(best_path)
    save_and_visualize(routes, best_dist, best_v, duration, config, df_locs)

def save_and_visualize(routes, dist, vehicles, time_cost, config, df_locs):
    result = {
        "solver_name": "ACO",
        "total_distance_km": dist / 100.0,
        "execution_time": time_cost,
        "routes": routes,
        "num_vehicles": vehicles
    }
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['output_dir'], 'ACO')
    os.makedirs(out_dir, exist_ok=True)
    ResultHandler.save_to_txt(result, out_dir)
    
    # Visualize
    vis = Visualizer(df_locs, osrm_url=config.get('visualization', {}).get('osrm_url', "http://localhost:5001"))
    vis.draw(routes, os.path.join(out_dir, 'route_map.html'))
    print(f"Done! Distance: {result['total_distance_km']:.2f}km | Time: {time_cost:.2f}s")

if __name__ == '__main__':
    run_aco_solver()