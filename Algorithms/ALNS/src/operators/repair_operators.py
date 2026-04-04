import numpy as np

def get_actual_vehicle_count(routes):
    """Đếm số xe thực sự có chở khách."""
    return len([r for r in routes if len(r) > 2])

def greedy_insertion(state, random_state):
    """Chèn tham lam: Tìm vị trí có chi phí tăng thêm thấp nhất."""
    repaired = state.copy()
    random_state.shuffle(repaired.unassigned)
    
    max_v = state.config['constraints'].get('max_vehicles', 200)
    nodes_to_process = list(repaired.unassigned)
    repaired.unassigned = []

    for node in nodes_to_process:
        best_cost = float('inf')
        best_pos = None
        node_demand = repaired.demands[node]
        
        # 1. Thử chèn vào các lộ trình hiện có
        for r_idx, route in enumerate(repaired.routes):
            # Kiểm tra tải trọng (Dùng route_loads để tối ưu tốc độ)
            current_load = repaired.route_loads[r_idx] if hasattr(repaired, 'route_loads') else sum(repaired.demands[n] for n in route)
            
            if current_load + node_demand <= repaired.capacity:
                for i in range(1, len(route)):
                    prev, nxt = route[i-1], route[i]
                    cost = (repaired.distance_matrix[prev, node] + 
                            repaired.distance_matrix[node, nxt] - 
                            repaired.distance_matrix[prev, nxt])
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (r_idx, i)
        
        # 2. Thực hiện chèn
        if best_pos is not None:
            r_idx, pos_idx = best_pos
            repaired.routes[r_idx].insert(pos_idx, node)
            # Cập nhật cache tải trọng nếu có
            if hasattr(repaired, 'route_loads'):
                repaired.route_loads[r_idx] += node_demand
        else:
            # Nếu không chèn được vào xe cũ, thử tạo xe mới
            if get_actual_vehicle_count(repaired.routes) < max_v:
                new_route = [0, node, 0]
                repaired.routes.append(new_route)
                if hasattr(repaired, 'route_loads'):
                    repaired.route_loads.append(node_demand)
            else:
                # Không thể chèn, trả về danh sách chưa gán để chịu penalty
                repaired.unassigned.append(node)
                
    return repaired

def regret_insertion(state, random_state):
    """Chèn hối tiếc (Regret-2): Ưu tiên chèn các nốt khó chèn nhất."""
    repaired = state.copy()
    max_v = state.config['constraints'].get('max_vehicles', 200)

    while repaired.unassigned:
        regret_costs = []
        
        for node in repaired.unassigned:
            node_demand = repaired.demands[node]
            insertion_costs = []
            
            for r_idx, route in enumerate(repaired.routes):
                current_load = repaired.route_loads[r_idx] if hasattr(repaired, 'route_loads') else sum(repaired.demands[n] for n in route)
                
                if current_load + node_demand <= repaired.capacity:
                    for i in range(1, len(route)):
                        prev, nxt = route[i-1], route[i]
                        cost = (repaired.distance_matrix[prev, node] + 
                                repaired.distance_matrix[node, nxt] - 
                                repaired.distance_matrix[prev, nxt])
                        insertion_costs.append((cost, r_idx, i))
            
            insertion_costs.sort(key=lambda x: x[0])
            
            if len(insertion_costs) >= 2:
                regret = insertion_costs[1][0] - insertion_costs[0][0]
                regret_costs.append((regret, node, insertion_costs[0]))
            elif len(insertion_costs) == 1:
                # Ưu tiên cực cao cho nốt chỉ còn 1 chỗ duy nhất để chèn
                regret_costs.append((1e6 + insertion_costs[0][0], node, insertion_costs[0]))
            else:
                # Không tìm được chỗ chèn
                regret_costs.append((-1, node, None))

        # Chọn nốt có độ hối tiếc lớn nhất để chèn trước
        regret_costs.sort(key=lambda x: x[0], reverse=True)
        _, best_node, best_insertion = regret_costs[0]
        
        repaired.unassigned.remove(best_node)
        
        if best_insertion is not None:
            cost, r_idx, pos_idx = best_insertion
            repaired.routes[r_idx].insert(pos_idx, best_node)
            if hasattr(repaired, 'route_loads'):
                repaired.route_loads[r_idx] += repaired.demands[best_node]
        else:
            if get_actual_vehicle_count(repaired.routes) < max_v:
                repaired.routes.append([0, best_node, 0])
                if hasattr(repaired, 'route_loads'):
                    repaired.route_loads.append(repaired.demands[best_node])
            else:
                # Chấp nhận không gán được nốt này
                pass 
                
    return repaired