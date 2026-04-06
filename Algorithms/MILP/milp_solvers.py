from pulp import *

def solve_acvrp_milp(matrix, demands, num_vehicles, capacity, timelimit=120):
    """
    Giải bài toán ACVRP bằng MILP (công thức Multi-Commodity Flow).

    Ma trận đầu vào: đơn vị mét (số nguyên, OSRM).
    Hàm mục tiêu tối thiểu tổng mét — caller chia 1000 để đổi sang km.

    Mô hình MCF:
    - x[i,j] ∈ {0,1}: xe có đi từ i đến j không
    - f[i,j] ≥ 0    : lượng hàng tích lũy xe mang trên cạnh (i,j)

    Ràng buộc bảo toàn luồng (đúng chiều):
      Depot: tổng flow xuất = total_demand, tổng flow nhập = total_demand
      Customer i: flow_in(i) - flow_out(i) = demand[i]
      (xe mang đầy hàng từ depot, giao dần, flow giảm dọc tuyến)
    """
    n = len(matrix)
    nodes = list(range(n))
    customers = list(range(1, n))
    total_demand = sum(demands[i] for i in customers)

    prob = LpProblem("ACVRP_Optimization", LpMinimize)

    valid_edges = [(i, j) for i in nodes for j in nodes if i != j]

    x = LpVariable.dicts("x", valid_edges, 0, 1, cat=LpBinary)
    f = LpVariable.dicts("f", valid_edges, 0, capacity * num_vehicles, cat=LpContinuous)

    # Hàm mục tiêu: tổng khoảng cách (đơn vị mét)
    prob += lpSum(matrix[i][j] * x[i, j] for (i, j) in valid_edges), "Total_Distance_m"

    # Ràng buộc bậc: mỗi khách hàng vào/ra đúng 1 lần
    for i in customers:
        prob += lpSum(x[j, i] for j in nodes if i != j) == 1, f"Enter_{i}"
        prob += lpSum(x[i, j] for j in nodes if i != j) == 1, f"Leave_{i}"

    # Số xe xuất phát ≤ K
    prob += lpSum(x[0, j] for j in customers) <= num_vehicles, "Max_Vehicles"

    # FIX: Ràng buộc bảo toàn luồng tại depot
    # Flow xuất depot = total_demand (xe mang đầy hàng ra)
    prob += lpSum(f[0, j] for j in customers) == total_demand, "Depot_Flow_Out"
    # Flow nhập depot = total_demand (xe quay về sau khi giao hết)
    # FIX: Depot_Flow_In phải = total_demand, KHÔNG phải = 0
    # (Lỗi cũ: == 0 khiến xe không thể quay về depot → infeasible)
    prob += lpSum(f[j, 0] for j in customers) == total_demand, "Depot_Flow_In"

    # FIX: Bảo toàn luồng qua khách hàng (chiều đúng)
    # flow_in(i) - flow_out(i) = demand[i]
    # Giải thích: xe đến i mang flow_in hàng, giao demand[i], rời đi với flow_out = flow_in - demand[i]
    for i in customers:
        flow_in  = lpSum(f[j, i] for j in nodes if i != j)
        flow_out = lpSum(f[i, j] for j in nodes if i != j)
        prob += flow_in - flow_out == demands[i], f"FlowConserve_{i}"

    # Liên kết f và x: f[i,j] > 0 chỉ khi x[i,j] = 1
    for (i, j) in valid_edges:
        prob += f[i, j] <= capacity * num_vehicles * x[i, j], f"Link_{i}_{j}"

    # Giải
    solver = PULP_CBC_CMD(timeLimit=timelimit, msg=1)
    status = prob.solve(solver)
    status_str = LpStatus[status]
    obj_val = value(prob.objective)

    if status == -1 or obj_val is None:
        print(f"[MILP] Không tìm thấy nghiệm khả thi. Status: {status_str}")
        return status_str, None, []

    routes_info = _extract_routes(x, nodes, customers, demands, capacity)
    return status_str, obj_val, routes_info


def _extract_routes(x, nodes, customers, demands, capacity):
    """
    Truy vết các tuyến từ nghiệm x[i,j].
    Giới hạn max_steps để tránh vòng lặp vô tận khi timelimit khiến nghiệm xấp xỉ.
    """
    routes_info = []
    visited_nodes = set()
    max_steps = len(customers) + 1

    for j in customers:
        val_0j = value(x.get((0, j), 0))
        if val_0j is None or val_0j <= 0.5:
            continue
        if j in visited_nodes:
            continue

        route = [0, j]
        current_load = demands.get(j, 0)
        visited_nodes.add(j)
        curr = j
        steps = 0

        while curr != 0 and steps < max_steps:
            steps += 1
            next_node = None
            best_val = 0.5

            for nxt in nodes:
                if curr == nxt:
                    continue
                val = value(x.get((curr, nxt), 0))
                if val is None or val <= best_val:
                    continue
                if nxt in visited_nodes and nxt != 0:
                    print(f"[CẢNH BÁO] Chu trình con tại node {nxt}, cắt ngắn tuyến.")
                    next_node = 0
                    best_val = val
                    break
                next_node = nxt
                best_val = val

            if next_node is None:
                print(f"[CẢNH BÁO] Tuyến đứt tại node {curr}. Thêm depot để đóng.")
                route.append(0)
                break

            route.append(next_node)
            if next_node != 0:
                current_load += demands.get(next_node, 0)
                visited_nodes.add(next_node)
            curr = next_node

        if route[-1] != 0:
            route.append(0)

        is_valid = current_load <= capacity
        if not is_valid:
            print(f"[CẢNH BÁO] Tuyến {route} vi phạm capacity: load={current_load} > {capacity}")

        routes_info.append({
            'route': route,
            'load': current_load,
            'is_valid': is_valid
        })

    return routes_info
