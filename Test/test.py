import re
import os
import argparse
import pandas as pd


# =========================
# 1. PARSE ROUTES
# =========================
def extract_routes_from_text(file_path):
    """
    Trích xuất các route từ file txt.
    Hỗ trợ format:
    - Xe #001: 0 -> 5 -> 10 -> 0
    - Route #001: 0 -> 5 -> 10 -> 0 (cost = xxx)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    routes = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ("Xe #" in line or "Route #" in line) and "->" in line:
                try:
                    # Lấy phần sau dấu :
                    route_part = line.split(":")[-1].strip()

                    # Loại bỏ phần (cost = ...)
                    route_clean = route_part.split("(")[0].strip()

                    # Parse theo dấu ->
                    nodes = [int(x.strip()) for x in route_clean.split("->")]

                    if len(nodes) >= 2:
                        routes.append(nodes)

                except Exception as e:
                    print(f"[WARN] Lỗi parse dòng: {line.strip()} | {e}")

    return routes


# =========================
# 2. LOAD DISTANCE MATRIX
# =========================
def load_distance_matrix(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    return df.values


# =========================
# 3. VALIDATION
# =========================
def validate_routes(routes, matrix):
    n = len(matrix)
    visited = set()

    for i, route in enumerate(routes):
        # Check depot
        if route[0] != 0 or route[-1] != 0:
            print(f"[WARN] Route #{i} không bắt đầu/kết thúc tại depot: {route}")

        # Check index hợp lệ
        for node in route:
            if node < 0 or node >= n:
                raise ValueError(f"Node {node} vượt kích thước ma trận ({n})")

        # Collect visited (bỏ depot)
        visited.update(route[1:-1])

    # Check missing node
    expected = set(range(1, n))
    missing = expected - visited
    extra = visited - expected

    if missing:
        print(f"[WARN] Thiếu node: {sorted(missing)[:10]} ...")

    if extra:
        print(f"[WARN] Node dư: {sorted(extra)[:10]} ...")

    print(f"[INFO] Số node đã phục vụ: {len(visited)}/{n-1}")


# =========================
# 4. VERIFY DISTANCE
# =========================
def verify_optimization(routes, matrix, scaling=1.0, debug_limit=10):
    total_distance = 0

    print(f"{'Phương tiện':<15} | {'Số điểm':<10} | {'Quãng đường (km)':<15}")
    print("-" * 50)

    for i, route in enumerate(routes):
        route_dist = 0

        for j in range(len(route) - 1):
            u, v = route[j], route[j + 1]
            route_dist += float(matrix[u][v])

        dist_km = route_dist / scaling
        total_distance += dist_km

        if i < debug_limit:
            print(f"Route #{i:03d}       | {len(route)-2:<10} | {dist_km:>15.2f}")

    if len(routes) > debug_limit:
        print(f"... và {len(routes) - debug_limit} route khác.")

    return total_distance


# =========================
# 5. MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Verify VRP solution")

    parser.add_argument("--result", type=str, required=True,
                        help="File txt chứa kết quả route để kiểm tra")
    parser.add_argument("--matrix", type=str, required=True,
                        help="File CSV chứa ma trận khoảng cách")
    parser.add_argument("--scaling", type=float, default=1.0,
                        help="Scaling factor")
    parser.add_argument("--original", type=float, default=None,
                        help="Giá trị cost gốc để so sánh")

    args = parser.parse_args()

    print("=== VERIFY VRP SOLUTION ===")

    # 1. Extract routes
    routes = extract_routes_from_text(args.result)

    if not routes:
        print("[-] Không tìm thấy route.")
        return

    print(f"[+] Số route: {len(routes)}")

    # 2. Load matrix
    matrix = load_distance_matrix(args.matrix)
    print(f"[+] Kích thước ma trận: {matrix.shape}")

    # 3. Validate
    print("\n--- VALIDATION ---")
    validate_routes(routes, matrix)

    # 4. Verify distance
    print("\n--- RE-CALCULATE DISTANCE ---")
    recalculated = verify_optimization(routes, matrix, args.scaling)

    print("\n" + "=" * 50)
    print(f"TỔNG QUÃNG ĐƯỜNG: {recalculated:.4f} km")

    # 5. Compare với kết quả gốc
    if args.original is not None:
        gap = abs(recalculated - args.original)
        print(f"GAP so với kết quả gốc: {gap:.4f} km")

    print("=" * 50)


if __name__ == "__main__":
    main()


# Sample usage:
#python test.py \
#  --result ../Results/SA/result_sa.txt \
#  --matrix ../Data/orsm_matrix.csv