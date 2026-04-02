import os
import sys
import json
import time
import subprocess
import traceback

# ===== FIX ENCODING CHO WINDOWS CONSOLE =====
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ===== ĐƯỜNG DẪN GỐC =====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'Utils', 'config.json')

# ===== IMPORT UTILS CHUNG (cho PyVRP & OR-Tools chạy trực tiếp) =====
from Utils.Data_loader import DataLoader
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer
from Algorithms.ORTools.solver.solver_OR_Tools import ORToolsSolver
from Algorithms.PyVRP.solver.solver_pyVRP import PyVRPSolver

# =====================================================================
#  DANH SÁCH THUẬT TOÁN
# =====================================================================
ALGORITHMS = {
    1: {
        "name": "PyVRP (Hybrid Genetic Search)",
        "key": "py_vrp",
        "mode": "integrated"    # Chạy trực tiếp trong process này
    },
    2: {
        "name": "OR-Tools (Google CP Solver)",
        "key": "or_tools",
        "mode": "integrated"
    },
    3: {
        "name": "ACO (Ant Colony Optimization)",
        "key": "aco",
        "mode": "subprocess",   # Chạy bằng subprocess
        "script": os.path.join(PROJECT_ROOT, "Algorithms", "ACO", "solver_aco.py"),
        "cwd": os.path.join(PROJECT_ROOT, "Algorithms", "ACO")
    },
    4: {
        "name": "ALNS (Adaptive Large Neighborhood Search)",
        "key": "alns",
        "mode": "subprocess",
        "script": os.path.join(PROJECT_ROOT, "Algorithms", "ALNS", "main.py"),
        "cwd": os.path.join(PROJECT_ROOT, "Algorithms", "ALNS")
    },
    5: {
        "name": "LNS (Large Neighborhood Search)",
        "key": "lns",
        "mode": "subprocess",
        "script": os.path.join(PROJECT_ROOT, "Algorithms", "LNS", "main.py"),
        "cwd": os.path.join(PROJECT_ROOT, "Algorithms", "LNS")
    },
    6: {
        "name": "SA (Simulated Annealing)",
        "key": "sa",
        "mode": "subprocess",
        "script": os.path.join(PROJECT_ROOT, "Algorithms", "SA", "main.py"),
        "cwd": os.path.join(PROJECT_ROOT, "Algorithms", "SA")
    },
    7: {
        "name": "Tabu Search",
        "key": "tabu",
        "mode": "subprocess",
        "script": os.path.join(PROJECT_ROOT, "Algorithms", "Tabu", "main_tabu.py"),
        "cwd": os.path.join(PROJECT_ROOT, "Algorithms", "Tabu")
    },
    8: {
        "name": "MILP (Mixed Integer Linear Programming)",
        "key": "milp",
        "mode": "subprocess",
        "script": os.path.join(PROJECT_ROOT, "Algorithms", "MILP", "main.py"),
        "cwd": os.path.join(PROJECT_ROOT, "Algorithms", "MILP")
    },
}


def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy cấu hình tại: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# =====================================================================
#  CHẠY PyVRP / OR-Tools (tích hợp trực tiếp)
# =====================================================================
def run_integrated(solver_type):
    """
    Quy trình thống nhất cho PyVRP và OR-Tools:
    Load Data -> Solve -> Standardize -> Save & Visualize
    """
    config = load_config(CONFIG_PATH)
    loader = DataLoader(config)
    data_bundle = loader.load_data()

    solver_cfg = config['solvers'].get(solver_type, {})
    time_limit = solver_cfg.get('time_limit_seconds', 300)

    start_time = time.time()
    standardized_result = {}

    if solver_type == "py_vrp":
        solver = PyVRPSolver(data_bundle['distance_matrix'], config['global_constraints'])
        res = solver.solve(time_limit=time_limit)

        routes_dict = {}
        for i, route in enumerate(res.best.routes()):
            routes_dict[i] = [0] + route.visits() + [0]

        total_distance = res.best.distance()
        scaling_factor = config['common_model_parameters']['scaling_factor']
        standardized_result = {
            "solver_name": "PyVRP",
            "total_distance_km": total_distance / scaling_factor,
            "execution_time": time.time() - start_time,
            "routes": routes_dict,
            "num_vehicles": len(routes_dict)
        }

    elif solver_type == "or_tools":
        solver = ORToolsSolver(data_bundle, config)
        routes, dist_km = solver.solve()

        standardized_result = {
            "solver_name": "OR-Tools",
            "total_distance_km": dist_km,
            "execution_time": time.time() - start_time,
            "routes": routes,
            "num_vehicles": len(routes)
        }

    # === HẬU XỬ LÝ ===
    if not standardized_result.get("routes"):
        print(f"(!) {solver_type} không tìm thấy lời giải.")
        return

    output_base = config['paths']['output_dir']
    solver_output_dir = os.path.join(output_base, solver_type)
    os.makedirs(solver_output_dir, exist_ok=True)

    ResultHandler.save_to_txt(standardized_result, solver_output_dir)
    ResultHandler.save_to_json(standardized_result, solver_output_dir)

    print(f"--- Đang trực quan hóa lộ trình ---")
    try:
        viz = Visualizer(
            data_bundle['df_locations'],
            osrm_url=config['visualization'].get('osrm_url', "http://localhost:5001"),
            use_osrm=config['visualization'].get('use_osrm', True)
        )
        map_path = os.path.join(solver_output_dir, config['visualization']['map_filename'])
        viz.draw(standardized_result['routes'], map_path)
        print(f"[HOÀN TẤT] Map lưu tại: {map_path}")
    except Exception as e:
        print(f"[WARNING] Trực quan hóa thất bại: {e}")

    print(f"\n[HOÀN TẤT] Kết quả {solver_type} lưu tại: {solver_output_dir}")


# =====================================================================
#  CHẠY CÁC THUẬT TOÁN KHÁC (subprocess)
# =====================================================================
def run_subprocess(algo_info):
    """
    Chạy thuật toán bằng subprocess với cwd đúng thư mục.
    Giữ nguyên 100% code gốc của thuật toán.
    """
    script = algo_info["script"]
    cwd = algo_info["cwd"]

    if not os.path.exists(script):
        print(f"[LỖI] Không tìm thấy script: {script}")
        return

    print(f"--- Đang chạy: {script} ---")
    print(f"--- Working dir: {cwd} ---\n")

    # Chạy subprocess, output stream trực tiếp ra console
    result = subprocess.run(
        [sys.executable, script],
        cwd=cwd,
        # Không capture output -> hiện trực tiếp ra terminal
    )

    if result.returncode == 0:
        print(f"\n[HOÀN TẤT] Thuật toán chạy thành công!")
    else:
        print(f"\n[LỖI] Thuật toán kết thúc với mã lỗi: {result.returncode}")


# =====================================================================
#  MENU CHÍNH
# =====================================================================
def show_menu():
    print("\n" + "=" * 60)
    print("  ACVRP HCMC - HỆ THỐNG TỐI ƯU HÓA ĐỊNH TUYẾN XE")
    print("=" * 60)
    print("\nChọn thuật toán muốn chạy:\n")

    for num, algo in ALGORITHMS.items():
        print(f"  [{num}] {algo['name']}")

    print(f"\n  [0] Thoát")
    print(f"  [9] Chạy TẤT CẢ (trừ MILP)")
    print("-" * 60)


def main():
    while True:
        show_menu()

        try:
            choice = input("\nNhập số thuật toán: ").strip()

            if choice == '0':
                print("Tạm biệt!")
                break

            if choice == '9':
                # Chạy tất cả (trừ MILP vì giới hạn 350 nodes)
                print("\n>>> CHẠY TẤT CẢ THUẬT TOÁN <<<\n")
                for num, algo in ALGORITHMS.items():
                    if algo['key'] == 'milp':
                        print(f"\n[SKIP] {algo['name']} (giới hạn 350 điểm, chạy riêng)")
                        continue
                    if algo['key'] == 'lns':
                        print(f"\n[SKIP] {algo['name']} (chưa có main.py)")
                        continue

                    print(f"\n{'='*20} {algo['name']} {'='*20}")
                    try:
                        if algo['mode'] == 'integrated':
                            run_integrated(algo['key'])
                        else:
                            run_subprocess(algo)
                    except Exception as e:
                        print(f"[LỖI] {algo['name']}: {e}")
                        traceback.print_exc()

                print("\n>>> HOÀN TẤT TẤT CẢ <<<")
                continue

            choice = int(choice)

            if choice not in ALGORITHMS:
                print("[!] Lựa chọn không hợp lệ. Vui lòng thử lại.")
                continue

            algo = ALGORITHMS[choice]
            print(f"\n{'='*20} KHỞI CHẠY: {algo['name']} {'='*20}")

            try:
                if algo['mode'] == 'integrated':
                    run_integrated(algo['key'])
                else:
                    run_subprocess(algo)
            except Exception as e:
                print(f"\n[LỖI HỆ THỐNG]: {str(e)}")
                traceback.print_exc()

        except ValueError:
            print("[!] Vui lòng nhập một số hợp lệ.")
        except KeyboardInterrupt:
            print("\n\nĐã dừng bởi người dùng.")
            break


if __name__ == "__main__":
    main()
