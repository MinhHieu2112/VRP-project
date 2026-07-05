# File chạy chính cho thuật toán MILP (Mixed-Integer Linear Programming) giải bài toán VRP sử dụng AlgorithmRunner.
from __future__ import annotations

import os
import sys
import time

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from milp_solvers import solve_acvrp_milp
from Algorithms.Init_strategies.Init_strategies import init_solution
from Utils.Pipeline import AlgorithmRunner, KM_SCALE


def compute_upper_bound(matrix, demands_dict: dict, capacity: int, max_vehicles: int, strategy: str = 'greedy') -> float:
    # Tính giới hạn trên cho hàm mục tiêu để định hướng solver MILP.
    num_nodes = matrix.shape[0]
    print(f"[MILP] Tính upper bound bằng chiến lược: '{strategy}'")
    solution = init_solution(strategy=strategy, matrix=matrix, num_nodes=num_nodes, capacity=capacity, demands=demands_dict, max_vehicles=max_vehicles, validate=False)
    ub_units = sum(sum(matrix[r[i], r[i + 1]] for i in range(len(r) - 1)) for r in solution)
    print(f"[MILP] Upper bound ({strategy}): {ub_units / KM_SCALE:.2f} km | {len(solution)} xe")
    return float(ub_units)


def format_routes(routes_info: list) -> dict:
    # Chuyển đổi thông tin tuyến đường từ MILP sang định dạng dict chuẩn.
    routes = {}
    for idx, info in enumerate(routes_info):
        route = info['route']
        if route[-1] != 0:
            route.append(0)
        routes[idx] = route
    return routes


class MILPSolverWrapper:
    """Wrapper bọc tiến trình tối ưu hóa MILP để tuân thủ interface solve() của AlgorithmRunner."""

    def __init__(self, data: dict, config: dict, limit_nodes: int = 200) -> None:
        self.data = data
        self.config = config
        self.limit_nodes = limit_nodes

    def solve(self) -> tuple[dict, float]:
        matrix      = self.data['distance_matrix'][:self.limit_nodes, :self.limit_nodes]
        capacity    = self.data['vehicle_capacity']
        demands_arr = self.data['demands'][:self.limit_nodes]
        num_nodes   = matrix.shape[0]

        demands_dict = {i: int(demands_arr[i]) for i in range(num_nodes)}
        cons      = self.config.get('global_constraints', {})
        milp_cfg  = self.config.get('solvers', {}).get('milp', {})
        max_v     = cons.get('max_vehicles', 200)
        timelimit = milp_cfg.get('max_runtime_seconds', 300)
        ub_strategy = milp_cfg.get('upper_bound_strategy', 'greedy')

        print(f"[MILP] {num_nodes} node | {max_v} xe | capacity={capacity} | timelimit={timelimit}s")
        if num_nodes > 80:
            print(f"[MILP][WARN] MTZ formulation có O(n²) ràng buộc. n={num_nodes} → {num_nodes**2:,} MTZ constraints.")

        compute_upper_bound(matrix, demands_dict, capacity, max_v, strategy=ub_strategy)

        print("--- Đang giải bằng MILP (PuLP/CBC) ---")
        status_str, obj_val_units, routes_info = solve_acvrp_milp(matrix, demands_dict, num_vehicles=max_v, capacity=capacity, timelimit=timelimit)

        print(f"\nTrạng thái solver: {status_str}")

        if obj_val_units is None or not routes_info:
            raise ValueError(f"[MILP] Solver báo không tìm được nghiệm khả thi hoặc không truy vết được routes.")

        routes = format_routes(routes_info)
        return routes, obj_val_units


class MILPRunner(AlgorithmRunner):
    """Runner cho thuật toán MILP kế thừa AlgorithmRunner."""

    def __init__(self, name: str, config_path: str, subfolder: str | None = None, limit_nodes: int = 200) -> None:
        super().__init__(name, config_path, subfolder)
        self.limit_nodes = limit_nodes

    def build_solver(self, data: dict, config: dict) -> MILPSolverWrapper:
        # Khởi tạo wrapper cho MILP Solver.
        return MILPSolverWrapper(data, config, self.limit_nodes)


if __name__ == "__main__":
    runner = MILPRunner(
        name        = "MILP",
        config_path = os.path.join(CURRENT_DIR, "config.json"),
        limit_nodes = 200,
    )
    runner.run()