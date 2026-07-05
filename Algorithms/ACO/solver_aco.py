# File điều phối chính chạy thuật toán ACO (Ant Colony Optimization) giải bài toán VRP sử dụng AlgorithmRunner.
from __future__ import annotations

import os
import sys
import numpy as np

_THIS_DIR     = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Models.cvrp_base import CVRPGraph, Node
from Core.engine import BasicACO
from Algorithms.Init_strategies.Init_strategies import init_solution
from Utils.Pipeline import AlgorithmRunner, KM_SCALE


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


class ACOSolverWrapper:
    """Wrapper bọc tiến trình ACO engine để tuân thủ interface solve() của AlgorithmRunner."""

    def __init__(self, aco: BasicACO) -> None:
        self.aco = aco

    def solve(self) -> tuple[dict, float]:
        best_path, best_dist, _ = self.aco.run_basic_aco()
        routes = parse_routes(best_path)
        return routes, best_dist


class ACORunner(AlgorithmRunner):
    """Runner cho thuật toán ACO kế thừa AlgorithmRunner."""

    def build_solver(self, data: dict, config: dict) -> ACOSolverWrapper:
        # Khởi tạo graph, nạp seed pheromone và thiết lập ACO engine.
        aco_cfg = config.get('solvers', {}).get('aco', {})
        matrix_int = data['distance_matrix']
        vehicle_capacity = data['vehicle_capacity']
        df_locs = data['df_locations']
        demands = data['demands']

        num_nodes_from_matrix = matrix_int.shape[0]
        df_locs_sub = df_locs.head(num_nodes_from_matrix)
        demands_sub = demands[:num_nodes_from_matrix]

        nodes = [
            Node(int(row['id']), float(row['lat']), float(row['lon']), float(demands_sub[idx]))
            for idx, row in df_locs_sub.iterrows()
        ]
        print(f"[ACO] Đã đồng bộ: {len(nodes)} nodes khớp với ma trận {matrix_int.shape}")

        graph = CVRPGraph(len(nodes), nodes, matrix_int.astype(np.float64), vehicle_capacity, rho=0.1, xi=0.01)

        seed_sol, seed_cost, seed_weight = _initialize_seed(matrix_int, len(nodes), vehicle_capacity, aco_cfg)
        graph.seed_pheromone(seed_sol, seed_weight=seed_weight)

        aco = setup_aco_engine(graph, aco_cfg)

        flat_seed_path = []
        for route in seed_sol:
            if not flat_seed_path:
                flat_seed_path.extend(route)
            else:
                flat_seed_path.extend(route[1:])

        aco.best_path_distance = float(seed_cost) 
        aco.best_path = flat_seed_path 

        return ACOSolverWrapper(aco)


if __name__ == '__main__':
    runner = ACORunner(
        name        = "ACO",
        config_path = os.path.join(_THIS_DIR, "config.json"),
    )
    runner.run()