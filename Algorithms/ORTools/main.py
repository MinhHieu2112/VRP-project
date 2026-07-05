# File chạy chính cho thuật toán OR-Tools (Guided Local Search) giải bài toán VRP.
import os
import sys
import json
import time

_THIS_FILE   = os.path.realpath(__file__)
_ALGO_DIR    = os.path.dirname(_THIS_FILE)
_ALGOS_DIR   = os.path.dirname(_ALGO_DIR)
PROJECT_ROOT = os.path.dirname(_ALGOS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Utils.Pipeline import load_data, build_result, save_result, visualize
from Algorithms.ORTools.solver.solver_OR_Tools import ORToolsSolver

CONFIG_PATH = os.path.join(_ALGO_DIR, 'config.json')


def load_config() -> dict:
    # Đọc thông tin cấu hình của OR-Tools từ tệp config.json.
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Không tìm thấy config: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # Thực hiện toàn bộ quy trình giải VRP bằng OR-Tools và lưu kết quả.
    print("=" * 60)
    print("  OR-Tools — Guided Local Search")
    print("=" * 60)

    config      = load_config()
    data_bundle = load_data(config)

    solver_cfg       = config.get("solvers", {}).get("or_tools", {})
    no_improve_iters = solver_cfg.get("no_improve_iters", 200)

    solver = ORToolsSolver(data_bundle, config)

    t0                  = time.time()
    routes, total_units = solver.solve(no_improve_iters=no_improve_iters)
    elapsed             = time.time() - t0

    if routes is None:
        print("[!] OR-Tools không tìm thấy lời giải.")
        sys.exit(1)

    result = build_result("OR-Tools", routes, total_units, elapsed)

    print(f"\n[KẾT QUẢ] Tổng quãng đường : {result['total_distance_km']:.2f} km")
    print(f"[KẾT QUẢ] Số xe sử dụng    : {result['num_vehicles']}")
    print(f"[KẾT QUẢ] Thời gian chạy   : {elapsed:.2f}s")

    save_result(result, config, subfolder="or_tools")
    visualize(result, config, subfolder="or_tools", df_locations=data_bundle["df_locations"])

    print(f"\n[HOÀN TẤT] Kết quả đã lưu tại: Results/or_tools/")


if __name__ == "__main__":
    main()