# File chạy chính cho thuật toán PyVRP (Hybrid Genetic Search) giải bài toán VRP.
import os
import sys
import json
import time

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

from Utils.Pipeline import load_data, build_result, save_result, visualize
from Algorithms.PyVRP.solver.solver_pyVRP import PyVRPSolver

CONFIG_PATH = os.path.join(_ALGO_DIR, 'config.json')


def load_config() -> dict:
    # Đọc thông tin cấu hình của PyVRP từ tệp config.json.
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Không tìm thấy config: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # Thực hiện toàn bộ quy trình giải VRP bằng PyVRP và lưu kết quả.
    print("=" * 60)
    print("  PyVRP — Hybrid Genetic Search")
    print("=" * 60)

    config      = load_config()
    data_bundle = load_data(config)

    solver_cfg       = config.get("solvers", {}).get("py_vrp", {})
    no_improve_iters = solver_cfg.get("no_improve_iters", 40000)
    display_log      = solver_cfg.get("display_log", True)

    solver = PyVRPSolver(
        matrix      = data_bundle["distance_matrix"],
        constraints = config["global_constraints"]
    )

    t0  = time.time()
    res = solver.solve(no_improve_iters=no_improve_iters, display=display_log)
    elapsed = time.time() - t0

    routes_dict = {}
    for i, route in enumerate(res.best.routes()):
        routes_dict[i] = [0] + route.visits() + [0]

    total_units = res.best.distance()
    result      = build_result("PyVRP", routes_dict, total_units, elapsed)

    print(f"\n[KẾT QUẢ] Tổng quãng đường : {result['total_distance_km']:.2f} km")
    print(f"[KẾT QUẢ] Số xe sử dụng    : {result['num_vehicles']}")
    print(f"[KẾT QUẢ] Thời gian chạy   : {elapsed:.2f}s")

    save_result(result, config, subfolder="py_vrp")
    visualize(result, config, subfolder="py_vrp", df_locations=data_bundle["df_locations"])

    print(f"\n[HOÀN TẤT] Kết quả đã lưu tại: Results/py_vrp/")


if __name__ == "__main__":
    main()