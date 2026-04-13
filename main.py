import os
import sys
import subprocess
import traceback

# ── Fix encoding Windows ──────────────────────────────────────────────────────
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
#  DANH SÁCH THUẬT TOÁN
#  Mỗi entry: tên hiển thị + đường dẫn script entry point
# =====================================================================
ALGORITHMS = {
    1: {
        "name":   "PyVRP (Hybrid Genetic Search)",
        "script": os.path.join(PROJECT_ROOT, "Algorithms", "PyVRP",  "main.py"),
        "cwd":    PROJECT_ROOT,
    },
    2: {
        "name":   "OR-Tools (Guided Local Search)",
        "script": os.path.join(PROJECT_ROOT, "Algorithms", "ORTools", "main.py"),
        "cwd":    PROJECT_ROOT,
    },
    3: {
        "name":   "ACO (Ant Colony Optimization)",
        "script": os.path.join(PROJECT_ROOT, "Algorithms", "ACO",    "solver_aco.py"),
        "cwd":    os.path.join(PROJECT_ROOT, "Algorithms", "ACO"),
    },
    4: {
        "name":   "ALNS (Adaptive Large Neighborhood Search)",
        "script": os.path.join(PROJECT_ROOT, "Algorithms", "ALNS",   "main.py"),
        "cwd":    os.path.join(PROJECT_ROOT, "Algorithms", "ALNS"),
    },
    5: {
        "name":   "SA (Simulated Annealing)",
        "script": os.path.join(PROJECT_ROOT, "Algorithms", "SA",     "main.py"),
        "cwd":    os.path.join(PROJECT_ROOT, "Algorithms", "SA"),
    },
    6: {
        "name":   "Tabu Search",
        "script": os.path.join(PROJECT_ROOT, "Algorithms", "Tabu",   "main_tabu.py"),
        "cwd":    os.path.join(PROJECT_ROOT, "Algorithms", "Tabu"),
    },
    7: {
        "name":   "MILP (Mixed Integer Linear Programming) [≤350 điểm]",
        "script": os.path.join(PROJECT_ROOT, "Algorithms", "MILP",   "main.py"),
        "cwd":    os.path.join(PROJECT_ROOT, "Algorithms", "MILP"),
    },
}

SKIP_ALL = {7}   # thuật toán bỏ qua khi chạy "tất cả"


# =====================================================================
#  DISPATCH
# =====================================================================
def run(algo: dict) -> None:
    """Chạy một thuật toán bằng subprocess, output stream trực tiếp ra terminal."""
    script = algo["script"]
    cwd    = algo["cwd"]

    if not os.path.exists(script):
        print(f"[LỖI] Không tìm thấy script: {script}")
        return

    print(f"\n--- Chạy: {script}")
    print(f"--- CWD : {cwd}\n")

    result = subprocess.run([sys.executable, script], cwd=cwd)

    if result.returncode == 0:
        print(f"\n[HOÀN TẤT] {algo['name']} chạy thành công.")
    else:
        print(f"\n[LỖI] {algo['name']} kết thúc với mã lỗi: {result.returncode}")


# =====================================================================
#  MENU
# =====================================================================
def show_menu() -> None:
    print("\n" + "=" * 60)
    print("  ACVRP HCMC — HỆ THỐNG TỐI ƯU HÓA ĐỊNH TUYẾN XE")
    print("=" * 60)
    print("\nChọn thuật toán:\n")
    for num, algo in ALGORITHMS.items():
        skip_note = "  ← bỏ qua khi chạy 'tất cả'" if num in SKIP_ALL else ""
        print(f"  [{num}] {algo['name']}{skip_note}")
    print(f"\n  [8] Chạy TẤT CẢ (trừ MILP)")
    print(f"  [0] Thoát")
    print("-" * 60)


def main() -> None:
    while True:
        show_menu()
        try:
            choice = input("\nNhập số thuật toán: ").strip()

            if choice == '0':
                print("Tạm biệt!")
                break

            if choice == '8':
                print("\n>>> CHẠY TẤT CẢ THUẬT TOÁN <<<\n")
                for num, algo in ALGORITHMS.items():
                    if num in SKIP_ALL:
                        print(f"\n[SKIP] {algo['name']}")
                        continue
                    print(f"\n{'='*20} {algo['name']} {'='*20}")
                    try:
                        run(algo)
                    except Exception as exc:
                        print(f"[LỖI] {algo['name']}: {exc}")
                        traceback.print_exc()
                print("\n>>> HOÀN TẤT TẤT CẢ <<<")
                continue

            num = int(choice)
            if num not in ALGORITHMS:
                print("[!] Lựa chọn không hợp lệ.")
                continue

            algo = ALGORITHMS[num]
            print(f"\n{'='*20} KHỞI CHẠY: {algo['name']} {'='*20}")
            run(algo)

        except ValueError:
            print("[!] Vui lòng nhập một số hợp lệ.")
        except KeyboardInterrupt:
            print("\n\nĐã dừng bởi người dùng.")
            break


if __name__ == "__main__":
    main()