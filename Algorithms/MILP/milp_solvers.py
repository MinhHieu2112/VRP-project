# milp_solvers.py — implement đúng theo mô hình MTZ trong báo cáo

from pulp import *

def solve_acvrp_milp(matrix, demands, num_vehicles, capacity, timelimit=120):
    """
    MILP cho ACVRP dùng công thức MTZ (Miller-Tucker-Zemlin),
    đúng theo mô hình toán trong báo cáo nghiên cứu (mục 2.1.5).

    Biến:
        x[i,j] ∈ {0,1}: xe đi từ i đến j
        u[i] ∈ [d_i, Q]: tải trọng tích lũy sau khi phục vụ i (biến MTZ)

    Ràng buộc:
        1. Mỗi khách hàng được vào đúng 1 lần
        2. Mỗi khách hàng được ra đúng 1 lần
        3. Số xe xuất phát và quay về ≤ K
        4. MTZ: u_i - u_j + Q*x_ij ≤ Q - d_j (loại subtour + kiểm soát tải)
        5. d_i ≤ u_i ≤ Q

    Ma trận đầu vào: km (float hoặc int đã scale)
    """
    n = len(matrix)
    nodes = list(range(n))
    customers = list(range(1, n))

    prob = LpProblem("ACVRP_MTZ", LpMinimize)

    # ── Biến nhị phân x[i,j] ──
    valid_edges = [(i, j) for i in nodes for j in nodes if i != j]
    x = LpVariable.dicts("x", valid_edges, cat=LpBinary)

    # ── Biến liên tục u[i]: tải trọng tích lũy sau khi phục vụ i ──
    # Chỉ cần cho customers (không cần cho depot)
    u = LpVariable.dicts("u", customers, lowBound=0, upBound=capacity, cat=LpContinuous)

    # ── Hàm mục tiêu ──
    prob += lpSum(matrix[i][j] * x[i, j] for (i, j) in valid_edges), "Total_Cost"

    # ── Ràng buộc 1: Mỗi khách hàng được VÀO đúng 1 lần ──
    for j in customers:
        prob += lpSum(x[i, j] for i in nodes if i != j) == 1, f"Enter_{j}"

    # ── Ràng buộc 2: Mỗi khách hàng được RA đúng 1 lần ──
    for i in customers:
        prob += lpSum(x[i, j] for j in nodes if i != j) == 1, f"Leave_{i}"

    # ── Ràng buộc 3: Số xe xuất phát và quay về ≤ K ──
    prob += lpSum(x[0, j] for j in customers) <= num_vehicles, "Depart_Max_K"
    prob += lpSum(x[i, 0] for i in customers) <= num_vehicles, "Return_Max_K"

    # ── Ràng buộc 4: MTZ — loại subtour + tải trọng ──
    # u_i - u_j + Q*x_ij ≤ Q - d_j  ∀i,j ∈ C, i≠j
    for i in customers:
        for j in customers:
            if i != j:
                prob += (
                    u[i] - u[j] + capacity * x[i, j] <= capacity - demands[j],
                    f"MTZ_{i}_{j}"
                )

    # ── Ràng buộc 5: Giới hạn tải trọng u_i ──
    # d_i ≤ u_i ≤ Q  ∀i ∈ C
    for i in customers:
        prob += u[i] >= demands[i], f"Load_LB_{i}"
        prob += u[i] <= capacity,   f"Load_UB_{i}"

    # ── Giải ──
    solver = PULP_CBC_CMD(timeLimit=timelimit, msg=1)
    status = prob.solve(solver)
    status_str = LpStatus[status]
    obj_val = value(prob.objective)

    if status not in (1,) and obj_val is None:
        print(f"[MILP] Không tìm được nghiệm. Status: {status_str}")
        return status_str, None, []

    routes_info = _extract_routes_mtz(x, nodes, customers, demands, capacity)
    return status_str, obj_val, routes_info


def _extract_routes_mtz(x, nodes, customers, demands, capacity):
    """Truy vết tuyến đường từ nghiệm x[i,j] của MTZ."""
    routes_info = []
    visited = set()
    max_steps = len(customers) + 1

    for j in customers:
        val = value(x.get((0, j), 0)) or 0
        if val <= 0.5 or j in visited:
            continue

        route = [0, j]
        load = demands.get(j, 0)
        visited.add(j)
        curr = j
        steps = 0

        while curr != 0 and steps < max_steps:
            steps += 1
            next_node = None
            best = 0.5
            for nxt in nodes:
                if curr == nxt:
                    continue
                v = value(x.get((curr, nxt), 0)) or 0
                if v > best:
                    if nxt in visited and nxt != 0:
                        next_node = 0
                        break
                    next_node = nxt
                    best = v
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