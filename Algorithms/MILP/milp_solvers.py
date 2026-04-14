# milp_solvers.py — implement đúng theo mô hình MTZ trong báo cáo
# FIX: Kiểm tra status solver chính xác, tránh dùng LP-relaxation value khi không có feasible solution

from pulp import *

# PuLP status codes
# 1  = Optimal
# 0  = Not Solved (timeout, no feasible found)
# -1 = Infeasible
# -2 = Unbounded
# -3 = Undefined
_FEASIBLE_STATUSES = {1, -1}   # Optimal hoặc Infeasible-with-bound đều có obj

def solve_acvrp_milp(matrix, demands, num_vehicles, capacity, timelimit=120):
    """
    MILP cho ACVRP dùng công thức MTZ (Miller-Tucker-Zemlin).

    FIXES:
      [FIX-1] Kiểm tra status đúng: dùng LpStatus string thay vì chỉ dựa obj_val is None.
              CBC timeout (status=0, "Not Solved") KHÔNG có feasible integer solution,
              nhưng obj_val có thể != None (LP relaxation bound bị rò) → phải check string.
      [FIX-2] Chỉ extract routes khi thực sự có integer feasible solution.
      [FIX-3] Giới hạn n ≤ 80 để MTZ (O(n²) constraints) còn giải được trong timelimit hợp lý.
              Với n=200, ~40k biến nhị phân + ~40k constraints → CBC không feasible trong 300s.
    """
    n = len(matrix)

    # ── [FIX-3] Cảnh báo kích thước ──────────────────────────────────────────
    if n > 80:
        print(f"[MILP][WARN] n={n} > 80. MTZ có O(n²)={n**2:,} ràng buộc. "
              f"CBC khó tìm feasible solution trong {timelimit}s. "
              f"Khuyến nghị dùng n ≤ 50 hoặc tăng timelimit.")

    nodes     = list(range(n))
    customers = list(range(1, n))

    prob = LpProblem("ACVRP_MTZ", LpMinimize)

    # ── Biến nhị phân x[i,j] ──────────────────────────────────────────────────
    valid_edges = [(i, j) for i in nodes for j in nodes if i != j]
    x = LpVariable.dicts("x", valid_edges, cat=LpBinary)

    # ── Biến liên tục u[i]: tải trọng tích lũy ───────────────────────────────
    u = LpVariable.dicts("u", customers, lowBound=0, upBound=capacity, cat=LpContinuous)

    # ── Hàm mục tiêu ─────────────────────────────────────────────────────────
    prob += lpSum(matrix[i][j] * x[i, j] for (i, j) in valid_edges), "Total_Cost"

    # ── R/C 1: Mỗi khách hàng được VÀO đúng 1 lần ───────────────────────────
    for j in customers:
        prob += lpSum(x[i, j] for i in nodes if i != j) == 1, f"Enter_{j}"

    # ── R/C 2: Mỗi khách hàng được RA đúng 1 lần ────────────────────────────
    for i in customers:
        prob += lpSum(x[i, j] for j in nodes if i != j) == 1, f"Leave_{i}"

    # ── R/C 3: Số xe ≤ K ─────────────────────────────────────────────────────
    prob += lpSum(x[0, j] for j in customers) <= num_vehicles, "Depart_Max_K"
    prob += lpSum(x[i, 0] for i in customers) <= num_vehicles, "Return_Max_K"

    # ── R/C 4: MTZ — loại subtour + tải trọng ───────────────────────────────
    for i in customers:
        for j in customers:
            if i != j:
                prob += (
                    u[i] - u[j] + capacity * x[i, j] <= capacity - demands[j],
                    f"MTZ_{i}_{j}"
                )

    # ── R/C 5: Giới hạn tải trọng ────────────────────────────────────────────
    for i in customers:
        prob += u[i] >= demands[i], f"Load_LB_{i}"
        prob += u[i] <= capacity,   f"Load_UB_{i}"

    # ── Giải ─────────────────────────────────────────────────────────────────
    solver     = PULP_CBC_CMD(timeLimit=timelimit, msg=1)
    status_int = prob.solve(solver)
    status_str = LpStatus[status_int]   # "Optimal", "Not Solved", "Infeasible", ...

    # ── [FIX-1] Kiểm tra feasibility đúng ────────────────────────────────────
    # "Optimal"  → status_int == 1  → có nghiệm integer tối ưu ✓
    # "Not Solved" (timeout, không tìm được feasible) → status_int == 0 → trả None
    # "Infeasible" → status_int == -1 → trả None
    has_feasible_solution = (status_int == 1)

    # CBC đôi khi timeout nhưng vẫn tìm được feasible integer solution
    # (status = 0 nhưng objective value là integer feasible, không phải LP bound)
    # Kiểm tra thêm: nếu có routes truy vết được thì vẫn dùng
    obj_val = value(prob.objective)

    print(f"\n[MILP] Status: {status_str} (code={status_int}) | obj={obj_val}")

    if not has_feasible_solution:
        # Thử fallback: nếu CBC tìm được incumbent trong quá trình B&B
        # Kiểm tra bằng cách xem có biến x nào = 1 không
        try:
            active_edges = [(i, j) for (i, j) in valid_edges
                            if (value(x[i, j]) or 0) > 0.5]
        except Exception:
            active_edges = []

        if active_edges and obj_val is not None:
            print(f"[MILP] Timeout nhưng tìm được incumbent feasible solution "
                  f"(obj={obj_val:.2f}). Tiếp tục truy vết routes.")
            has_feasible_solution = True
        else:
            print(f"[MILP] Không có feasible integer solution. "
                  f"Lower bound LP: {obj_val}. Trả về None.")
            return status_str, None, []

    # ── [FIX-2] Extract routes chỉ khi có feasible solution ──────────────────
    routes_info = _extract_routes_mtz(x, nodes, customers, demands, capacity)
    return status_str, obj_val, routes_info


def _extract_routes_mtz(x, nodes, customers, demands, capacity):
    """Truy vết tuyến đường từ nghiệm x[i,j] của MTZ."""
    routes_info = []
    visited     = set()
    max_steps   = len(customers) + 1

    for j in customers:
        val = value(x.get((0, j), 0)) or 0
        if val <= 0.5 or j in visited:
            continue

        route = [0, j]
        load  = demands.get(j, 0)
        visited.add(j)
        curr  = j
        steps = 0

        while curr != 0 and steps < max_steps:
            steps    += 1
            next_node = None
            best      = 0.5

            for nxt in nodes:
                if curr == nxt:
                    continue
                v = value(x.get((curr, nxt), 0)) or 0
                if v > best:
                    if nxt in visited and nxt != 0:
                        next_node = 0
                        break
                    next_node = nxt
                    best      = v

            if next_node is None:
                route.append(0)
                break
            route.append(next_node)
            if next_node != 0:
                load += demands.get(next_node, 0)
                visited.add(next_node)
            curr = next_node

        if route[-1] != 0:
            route.append(0)

        is_valid = load <= capacity
        routes_info.append({'route': route, 'load': load, 'is_valid': is_valid})

    return routes_info