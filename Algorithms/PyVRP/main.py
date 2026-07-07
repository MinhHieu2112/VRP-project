# File khởi chạy chính cho thuật toán PyVRP (Hybrid Genetic Search) sử dụng AlgorithmRunner chuẩn hóa.
from __future__ import annotations

import os
import sys

_THIS_FILE   = os.path.realpath(__file__)
_ALGO_DIR    = os.path.dirname(_THIS_FILE)
_ALGOS_DIR   = os.path.dirname(_ALGO_DIR)
PROJECT_ROOT = os.path.dirname(_ALGOS_DIR)

for _p in list(sys.path):
    if os.path.normcase(_p) == os.path.normcase(_ALGO_DIR):
        sys.path.remove(_p)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from Utils.Pipeline import AlgorithmRunner, load_data, build_result, save_result, visualize
from Algorithms.PyVRP.solver.solver_pyVRP import PyVRPSolver


class PyVRPRunner(AlgorithmRunner):
    """Runner đặc thù cho PyVRP với bước xử lý kết quả dạng đặc thù của thư viện."""

    def build_solver(self, data, config):
        # Khởi tạo PyVRPSolver từ ma trận khoảng cách và ràng buộc bài toán.
        return PyVRPSolver(
            matrix      = data["distance_matrix"],
            constraints = config["global_constraints"],
        )

    def run(self):
        # Override run để xử lý định dạng kết quả đặc thù của thư viện PyVRP.
        import time
        config      = self._load_config()
        data_bundle = load_data(config)

        solver_cfg       = config.get("solvers", {}).get("py_vrp", {})
        no_improve_iters = solver_cfg.get("no_improve_iters", 40000)
        display_log      = solver_cfg.get("display_log", True)

        solver = self.build_solver(data_bundle, config)

        t0  = time.time()
        res = solver.solve(no_improve_iters=no_improve_iters, display=display_log)
        elapsed = time.time() - t0

        routes_dict = {
            i: [0] + route.visits() + [0]
            for i, route in enumerate(res.best.routes())
        }
        total_units = res.best.distance()
        result      = build_result(self.name, routes_dict, total_units, elapsed)

        print(f"\n[KẾT QUẢ] Tổng quãng đường : {result['total_distance_km']:.2f} km")
        print(f"[KẾT QUẢ] Số xe sử dụng    : {result['num_vehicles']}")
        print(f"[KẾT QUẢ] Thời gian chạy   : {elapsed:.2f}s")

        save_result(result, config, self.subfolder)
        visualize(result, config, self.subfolder, data_bundle["df_locations"])

        self._print_summary(result, elapsed)
        return result


if __name__ == "__main__":
    print("=" * 60)
    print("  PyVRP — Hybrid Genetic Search")
    print("=" * 60)
    runner = PyVRPRunner(
        name        = "PyVRP",
        config_path = os.path.join(_ALGO_DIR, "config.json"),
        subfolder   = "py_vrp",
    )
    runner.run()