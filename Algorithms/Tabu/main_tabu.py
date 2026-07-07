# File khởi chạy chính cho thuật toán Granular Tabu Search sử dụng AlgorithmRunner chuẩn hóa.
from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR    = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Utils.Pipeline import AlgorithmRunner
from Algorithms.Init_strategies.Init_strategies import init_solution
from Algorithms.Tabu.solver import GranularTabuSearch


class _TabuSolverWrapper:
    """Wrapper mỏng bọc GranularTabuSearch để thống nhất interface solve() -> (routes, cost)."""

    def __init__(self, data: dict, config: dict) -> None:
        # Khởi tạo wrapper với dữ liệu bài toán và cấu hình tham số GTS.
        self._data   = data
        self._config = config

    def solve(self):
        # Khởi tạo nghiệm ban đầu và chạy GTS, trả về (routes, cost_units).
        data   = self._data
        config = self._config

        matrix      = data['distance_matrix']
        capacity    = data['vehicle_capacity']
        demands_arr = data['demands']
        num_nodes   = matrix.shape[0]

        demands_dict = {i: int(demands_arr[i]) for i in range(num_nodes)}
        tabu_p = config['tabu_parameters']
        cons   = config.get('constraints', config.get('global_constraints', {}))

        init_strategy = tabu_p.get('init_strategy', 'clarke_wright')
        print(f"[Tabu] Khởi tạo nghiệm bằng chiến lược: '{init_strategy}'")

        initial_state = init_solution(
            strategy     = init_strategy,
            matrix       = matrix,
            num_nodes    = num_nodes,
            capacity     = capacity,
            demands      = demands_dict,
            max_vehicles = cons.get('max_vehicles', 200),
            validate     = True,
        )

        served  = {n for r in initial_state for n in r if n != 0}
        missing = set(range(1, num_nodes)) - served
        if missing:
            print(f"[WARN] {len(missing)} KH chưa được phục vụ sau init!")
        else:
            print(f"[OK] Init: {len(initial_state)} xe, tất cả {num_nodes - 1} KH được phục vụ")

        solver = GranularTabuSearch(
            distance_matrix = matrix,
            demands         = demands_dict,
            capacity        = capacity,
            max_v           = cons.get('max_vehicles', 200),
            tabu_size       = tabu_p.get('tabu_size',        20),
            max_iter        = tabu_p.get('max_iterations', 3000),
            max_no_improve  = tabu_p.get('max_no_improve',  400),
            granular_beta   = tabu_p.get('granular_beta',   1.5),
            granular_k      = tabu_p.get('granular_k',       15),
            penalty_lambda  = tabu_p.get('penalty_lambda', None),
            penalty_h       = tabu_p.get('penalty_h',        20),
        )

        return solver.solve(initial_state)


class TabuRunner(AlgorithmRunner):
    """Runner đặc thù cho Granular Tabu Search, cần bước khởi tạo nghiệm ban đầu trước khi giải."""

    def build_solver(self, data, config):
        # Tạo wrapper bao bọc GTS để chuẩn hóa interface giải thuật.
        return _TabuSolverWrapper(data, config)


if __name__ == "__main__":
    print("\n===== RUN GRANULAR TABU SEARCH (Toth & Vigo, 2003) =====")
    runner = TabuRunner(
        name        = "Granular Tabu Search",
        config_path = os.path.join(_THIS_DIR, "config_tabu.json"),
        subfolder   = "Tabu",
    )
    runner.run()