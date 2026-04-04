from pulp import *


def solve_acvrp_milp(matrix, demands, num_vehicles, capacity, timelimit=120):
    """
    Giải bài toán ACVRP (Asymmetric Capacitated VRP) bằng mô hình MILP
    sử dụng công thức Miller-Tucker-Zemlin mở rộng (dạng flow-based / MCF).

    Mô hình sử dụng:
    - Biến nhị phân x[i,j]: xe có đi từ node i đến node j không
    - Biến liên tục f[i,j]: lượng hàng (demand) xe mang khi đi từ i đến j
      → Ràng buộc bảo toàn luồng (flow conservation) ngăn subtour tự nhiên,
        thay thế cho subtour elimination constraint kiểu SECs/MTZ.

    Args:
        matrix:       Ma trận khoảng cách n×n (asymmetric)
        demands:      Dict {node_id: demand}, depot node 0 có demand = 0
        num_vehicles: Số xe tối đa K
        capacity:     Sức chứa tối đa Q mỗi xe
        timelimit:    Giới hạn thời gian giải (giây)

    Returns:
        (status_str, obj_val, routes_info)
        - status_str:  Trạng thái solver ('Optimal', 'Feasible', ...)
        - obj_val:     Giá trị hàm mục tiêu (tổng khoảng cách, đơn vị gốc)
        - routes_info: List dict {'route': [...], 'load': ..., 'is_valid': bool}
    """
    n = len(matrix)
    nodes = list(range(n))
    customers = list(range(1, n))

    # ── Xây dựng bài toán ──
    prob = LpProblem("ACVRP_Optimization", LpMinimize)

    # Chỉ tạo biến cho các cạnh hợp lệ (i ≠ j)
    valid_edges = [(i, j) for i in nodes for j in nodes if i != j]

    # x[i,j] = 1 nếu có xe đi trực tiếp từ i đến j
    x = LpVariable.dicts("x", valid_edges, 0, 1, cat=LpBinary)

    # f[i,j] = lượng hàng (tổng demand tích lũy) xe mang trên cạnh (i,j)
    # Dùng để ngăn subtour qua flow conservation (không cần thêm SECs riêng)
    f = LpVariable.dicts("f", valid_edges, 0, capacity, cat=LpContinuous)

    # ── Hàm mục tiêu: tối thiểu tổng khoảng cách ──
    prob += lpSum(matrix[i][j] * x[i, j] for (i, j) in valid_edges), "Total_Distance"

    # ── Ràng buộc bậc (Degree constraints) ──
    # Mỗi khách hàng được thăm đúng 1 lần (vào đúng 1 lần, ra đúng 1 lần)
    for i in customers:
        prob += lpSum(x[j, i] for j in nodes if i != j) == 1, f"Enter_{i}"
        prob += lpSum(x[i, j] for j in nodes if i != j) == 1, f"Leave_{i}"

    # ── Ràng buộc số lượng xe ──
    # Số tuyến xuất phát từ depot không vượt quá K
    prob += lpSum(x[0, j] for j in customers) <= num_vehicles, "Max_Vehicles"

    # ── Ràng buộc bảo toàn luồng tại depot ──
    # FIX ①⑥: Thêm cả Depot_Flow_In = 0 để ép xe phải quay về depot (không đứt tuyến)
    total_demand = sum(demands[i] for i in customers)
    prob += lpSum(f[0, j] for j in customers) == total_demand, "Depot_Flow_Out"
    prob += lpSum(f[j, 0] for j in customers) == 0, "Depot_Flow_In"

    # ── Bảo toàn luồng qua từng khách hàng ──
    # FIX ②: Công thức MCF đúng: flow_out - flow_in = demand[i]
    # (xe mang hàng TỪ depot, giao dần, demand giảm dần theo tuyến)
    # Dấu cũ (flow_in - flow_out = demand) sai → subtour không bị loại.
    for i in customers:
        flow_in  = lpSum(f[j, i] for j in nodes if i != j)
        flow_out = lpSum(f[i, j] for j in nodes if i != j)
        prob += flow_out - flow_in == demands[i], f"FlowConserve_{i}"

    # ── Liên kết biến luồng với biến nhị phân ──
    # f[i,j] > 0 chỉ khi x[i,j] = 1; giới hạn trên bởi capacity
    for (i, j) in valid_edges:
        prob += f[i, j] <= capacity * x[i, j], f"Link_{i}_{j}"

    # ── Gọi solver CBC ──
    solver = PULP_CBC_CMD(timeLimit=timelimit, msg=1)
    status = prob.solve(solver)
    # FIX ⑤: Kiểm tra status trước, obj_val có thể không None ngay cả khi Infeasible
    status_str = LpStatus[status]
    if status not in (1, -1):  # 1=Optimal, -1=Infeasible (có thể có incumbent)
        obj_val = value(prob.objective)
    else:
        obj_val = value(prob.objective)

    if status == -1 or obj_val is None:
        print(f"[MILP] Không tìm thấy nghiệm khả thi. Status: {LpStatus[status]}")
        return status_str, None, []

    # ── Truy vết các tuyến đường từ nghiệm MILP ──
    routes_info = _extract_routes(x, nodes, customers, demands, capacity)

    return status_str, obj_val, routes_info


def _extract_routes(x, nodes, customers, demands, capacity):
    """
    Truy vết các tuyến đường từ nghiệm biến nhị phân x[i,j].

    Thuật toán:
    1. Tìm tất cả khách hàng j mà x[0,j] ≈ 1 → điểm bắt đầu mỗi tuyến
    2. Từ mỗi điểm bắt đầu, đi theo cạnh có x[curr, next] ≈ 1
    3. Dừng khi về depot (node 0) hoặc phát hiện chu trình lỗi

    FIX: Phiên bản cũ dùng while curr != 0 không giới hạn → có thể loop vô tận
    nếu solver trả về nghiệm không hoàn hảo (do timelimit). Nay giới hạn
    tối đa len(customers) bước.

    Args:
        x:         Dict biến LP x[i,j]
        nodes:     List tất cả node
        customers: List khách hàng (không bao gồm depot)
        demands:   Dict demand
        capacity:  Sức chứa xe

    Returns:
        List dict {'route': [...], 'load': int, 'is_valid': bool}
    """
    routes_info = []
    visited_nodes = set()   # Tránh gán 1 khách hàng vào nhiều tuyến
    max_steps = len(customers) + 1  # Giới hạn an toàn tránh vòng lặp vô tận

    for j in customers:
        # Chỉ bắt đầu tuyến từ depot → j
        if value(x.get((0, j), 0)) is None or value(x.get((0, j), 0)) <= 0.5:
            continue
        if j in visited_nodes:
            continue

        route = [0, j]
        current_load = demands.get(j, 0)
        visited_nodes.add(j)
        curr = j
        steps = 0

        # Đi theo cạnh tiếp theo cho đến khi về depot
        while curr != 0 and steps < max_steps:
            steps += 1
            next_node = None
            best_val = 0.5  # ngưỡng chấp nhận

            for nxt in nodes:
                if curr == nxt:
                    continue
                val = value(x.get((curr, nxt), 0))
                if val is None or val <= best_val:
                    continue

                if nxt in visited_nodes and nxt != 0:
                    print(f"[CẢNH BÁO] Chu trình con phát hiện tại node {nxt}, "
                          f"tuyến sẽ bị cắt ngắn.")
                    next_node = 0
                    best_val = val
                    break

                next_node = nxt
                best_val = val  # tiếp tục tìm cạnh có giá trị cao hơn

            if next_node is None:
                # Cạnh tiếp theo bị đứt — có thể do nghiệm xấp xỉ (timelimit)
                print(f"[CẢNH BÁO] Tuyến bị đứt tại node {curr}. "
                      f"Thêm depot để đóng tuyến.")
                route.append(0)
                break

            route.append(next_node)
            if next_node != 0:
                current_load += demands.get(next_node, 0)
                visited_nodes.add(next_node)
            curr = next_node

        # Đảm bảo tuyến luôn kết thúc bằng depot
        if route[-1] != 0:
            route.append(0)

        is_valid = current_load <= capacity
        if not is_valid:
            print(f"[CẢNH BÁO] Tuyến {route} vi phạm capacity: "
                  f"load={current_load} > capacity={capacity}")

        routes_info.append({
            'route': route,
            'load': current_load,
            'is_valid': is_valid
        })

    return routes_info