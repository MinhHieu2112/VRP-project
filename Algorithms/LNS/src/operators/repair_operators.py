import numpy as np

def get_actual_vehicle_count(routes):
    """Đếm số xe thực sự có chở khách."""
    return len([r for r in routes if len(r) > 2])

def greedy_insertion(state, random_state):
    repaired = state.copy()
    random_state.shuffle(repaired.unassigned)
    
    # Lấy giới hạn xe từ config (giả sử bạn lưu trong state)
    max_v = getattr(state, 'max_vehicles', 200) 
    
    nodes_to_process = list(repaired.unassigned)
    repaired.unassigned = []

    for node in nodes_to_process:
        best_cost = float('inf')
        best_pos = None
        node_demand = repaired.demands[node]
        
        for r_idx, route in enumerate(repaired.routes):
            if repaired.get_route_load(route) + node_demand <= repaired.capacity:
                for i in range(1, len(route)):
                    cost = (repaired.distance_matrix[route[i-1], node] + 
                            repaired.distance_matrix[node, route[i]] - 
                            repaired.distance_matrix[route[i-1], route[i]])
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (r_idx, i)
        
        if best_pos:
            r_idx, pos_idx = best_pos
            repaired.routes[r_idx].insert(pos_idx, node)
        else:
            # KIỂM TRA GIỚI HẠN 200 XE
            if get_actual_vehicle_count(repaired.routes) < max_v:
                repaired.routes.append([0, node, 0])
            else:
                # Nếu hết xe, nốt này tạm thời không được gán
                repaired.unassigned.append(node)
                
    return repaired

def regret_insertion(state, random_state):
    repaired = state.copy()
    max_v = getattr(state, 'max_vehicles', 200)
    
    while repaired.unassigned:
        regret_costs = []
        
        for node in repaired.unassigned:
            insertion_costs = []
            node_demand = repaired.demands[node]
            
            for r_idx, route in enumerate(repaired.routes):
                if repaired.get_route_load(route) + node_demand <= repaired.capacity:
                    for i in range(1, len(route)):
                        cost = (repaired.distance_matrix[route[i-1], node] + 
                                repaired.distance_matrix[node, route[i]] - 
                                repaired.distance_matrix[route[i-1], route[i]])
                        insertion_costs.append((cost, r_idx, i))
            
            insertion_costs.sort(key=lambda x: x[0])
            
            if len(insertion_costs) >= 2:
                regret = insertion_costs[1][0] - insertion_costs[0][0]
                regret_costs.append((regret, node, insertion_costs[0]))
            elif len(insertion_costs) == 1:
                # Trọng số lớn cho nốt chỉ có 1 lựa chọn
                regret_costs.append((1e6 + insertion_costs[0][0], node, insertion_costs[0]))
            else:
                # Trường hợp không chèn được vào xe cũ
                regret_costs.append((0, node, None))
                
        regret_costs.sort(key=lambda x: x[0], reverse=True)
        _, best_node, best_insertion = regret_costs[0]
        
        repaired.unassigned.remove(best_node)
        
        if best_insertion:
            _, r_idx, pos_idx = best_insertion
            repaired.routes[r_idx].insert(pos_idx, best_node)
        else:
            # KIỂM TRA GIỚI HẠN 200 XE
            if get_actual_vehicle_count(repaired.routes) < max_v:
                repaired.routes.append([0, best_node, 0])
            else:
                # Không thể tạo thêm xe, nốt này bị bỏ lại
                # Lưu ý: ALNS sẽ tính phạt Objective cho nốt unassigned này
                repaired.unassigned.append(best_node)
                break # Ngừng vòng lặp vì không còn chỗ chèn
            
    return repaired