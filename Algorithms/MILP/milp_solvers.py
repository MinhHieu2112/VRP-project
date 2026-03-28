from pulp import *

def solve_acvrp_milp(matrix, demands, num_vehicles, capacity, timelimit=120):
    """
    Hàm lõi chạy mô hình MILP cho bài toán ACVRP.
    Trả về: (status_string, objective_value, routes_info)
    """
    n = len(matrix)
    nodes = list(range(n))
    customers = list(range(1, n))
    
    prob = LpProblem("ACVRP_Optimization", LpMinimize)

    # KHẮC PHỤC 1: Chỉ tạo biến cho các cạnh hợp lệ (i != j)
    valid_edges = [(i, j) for i in nodes for j in nodes if i != j]
    
    x = LpVariable.dicts("x", valid_edges, 0, 1, cat=LpBinary)
    f = LpVariable.dicts("f", valid_edges, 0, capacity, cat=LpContinuous)

    # Hàm mục tiêu
    prob += lpSum([matrix[i][j] * x[i, j] for (i, j) in valid_edges])

    # Ràng buộc bậc (Degree constraints)
    for i in customers:
        prob += lpSum([x[j, i] for j in nodes if i != j]) == 1
        prob += lpSum([x[i, j] for j in nodes if i != j]) == 1

    # Ràng buộc số lượng xe
    prob += lpSum([x[0, j] for j in customers]) <= num_vehicles

    # KHẮC PHỤC 2: Ràng buộc tổng luồng rời Depot phải bằng tổng nhu cầu
    prob += lpSum([f[0, j] for j in customers]) == sum([demands[i] for i in customers])

    # Bảo toàn luồng qua từng khách hàng
    for i in customers:
        prob += lpSum([f[j, i] for j in nodes if i != j]) - \
                lpSum([f[i, j] for j in nodes if i != j]) == demands[i]

    # Liên kết biến luồng và biến nhị phân
    for (i, j) in valid_edges:
        prob += f[i, j] <= capacity * x[i, j]

    # Gọi solver
    status = prob.solve(PULP_CBC_CMD(timeLimit=timelimit, msg=0))
    status_str = LpStatus[status]

    # KHẮC PHỤC 5: Kiểm tra nghiệm khả thi một cách an toàn
    obj_val = value(prob.objective)
    if obj_val is None:
        return status_str, None, []

    # KHẮC PHỤC 6 & 7: Truy vết an toàn sử dụng visited set
    routes_info = []
    visited_nodes = set()
    
    for j in customers:
        if value(x[0, j]) is not None and value(x[0, j]) > 0.5:
            route = [0, j]
            curr = j
            current_route_load = demands[j]
            visited_nodes.add(j)
            
            while curr != 0:
                next_node_found = False
                for next_node in nodes:
                    if curr != next_node and value(x[curr, next_node]) is not None and value(x[curr, next_node]) > 0.5:
                        if next_node in visited_nodes and next_node != 0:
                            print(f"[Cảnh báo] Phát hiện chu trình con kẹt tại node {next_node}!")
                            break # Thoát để tránh lặp vô tận
                        
                        route.append(next_node)
                        if next_node != 0:
                            current_route_load += demands[next_node]
                            visited_nodes.add(next_node)
                        
                        curr = next_node
                        next_node_found = True
                        break
                
                if not next_node_found: 
                    break # Tuyến đường bị đứt gãy
            
            # KHẮC PHỤC 7: Kiểm tra vi phạm sức chứa
            is_valid = current_route_load <= capacity
            routes_info.append({
                'route': route,
                'load': current_route_load,
                'is_valid': is_valid
            })

    return status_str, obj_val, routes_info