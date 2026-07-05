# File khởi chạy chính cho thuật toán OR-Tools (Guided Local Search) sử dụng AlgorithmRunner chuẩn hóa.
from __future__ import annotations

import os
import sys

_THIS_FILE   = os.path.realpath(__file__)
_ALGO_DIR    = os.path.dirname(_THIS_FILE)
_ALGOS_DIR   = os.path.dirname(_ALGO_DIR)
PROJECT_ROOT = os.path.dirname(_ALGOS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Utils.Pipeline import AlgorithmRunner, load_data, build_result, save_result, visualize
from Algorithms.ORTools.solver.solver_OR_Tools import ORToolsSolver


class ORToolsRunner(AlgorithmRunner):
    """Runner đặc thù cho OR-Tools, thêm xử lý khi solver không tìm được nghiệm."""

    def build_solver(self, data, config):
        # Khởi tạo ORToolsSolver từ dữ liệu và cấu hình được nạp sẵn bởi AlgorithmRunner.
        return ORToolsSolver(data, config)

    def run(self):
        # Override run để xử lý trường hợp đặc biệt khi OR-Tools không tìm được lời giải.
        import sys
        config      = self._load_config()
        data_bundle = load_data(config)

        solver_cfg       = config.get("solvers", {}).get("or_tools", {})
        no_improve_iters = solver_cfg.get("no_improve_iters", 200)

        solver = self.build_solver(data_bundle, config)

        import time
        t0                  = time.time()
        routes, total_units = solver.solve(no_improve_iters=no_improve_iters)
        elapsed             = time.time() - t0

        if routes is None:
            print("[!] OR-Tools không tìm thấy lời giải.")
            sys.exit(1)

        result = build_result(self.name, routes, total_units, elapsed)

        print(f"\n[KẾT QUẢ] Tổng quãng đường : {result['total_distance_km']:.2f} km")
        print(f"[KẾT QUẢ] Số xe sử dụng    : {result['num_vehicles']}")
        print(f"[KẾT QUẢ] Thời gian chạy   : {elapsed:.2f}s")

        save_result(result, config, self.subfolder)
        visualize(result, config, self.subfolder, data_bundle["df_locations"])

        self._print_summary(result, elapsed)
        return result


if __name__ == "__main__":
    print("=" * 60)
    print("  OR-Tools — Guided Local Search")
    print("=" * 60)
    runner = ORToolsRunner(
        name        = "OR-Tools",
        config_path = os.path.join(_ALGO_DIR, "config.json"),
        subfolder   = "or_tools",
    )
    runner.run()