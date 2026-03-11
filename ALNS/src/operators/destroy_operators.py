import random
import numpy as np

def get_nodes_to_remove(state):
    """Xác định số lượng nút cần xóa (thường từ 5-15% tổng số điểm)."""
    num_nodes = len(state.distance_matrix) - 1 # Trừ kho
    return int(num_nodes * random.uniform(0.05, 0.1))

def random_removal(state, random_state):
    """Xóa ngẫu nhiên các khách hàng."""
    destroyed = state.copy()
    nodes_to_remove = get_nodes_to_remove(state)
    
    # Lấy tất cả khách hàng hiện có trong các route (trừ kho 0)
    all_clients = []
    for route in destroyed.routes:
        all_clients.extend([node for node in route if node != 0])
    
    if not all_clients:
        return destroyed

    # Chọn ngẫu nhiên n khách hàng để xóa
    to_remove = random_state.choice(all_clients, nodes_to_remove, replace=False)
    
    for node in to_remove:
        destroyed.unassigned.append(node)
        for route in destroyed.routes:
            if node in route:
                route.remove(node)
                
    # Dọn dẹp các route trống (chỉ còn [0, 0])
    destroyed.routes = [r for r in destroyed.routes if len(r) > 2]
    return destroyed

def worst_removal(state, random_state):
    """Xóa những khách hàng có chi phí đóng góp vào route cao nhất."""
    destroyed = state.copy()
    nodes_to_remove = get_nodes_to_remove(state)
    
    costs = []
    for route_idx, route in enumerate(destroyed.routes):
        for i in range(1, len(route) - 1):
            prev, node, next_node = route[i-1], route[i], route[i+1]
            # Chi phí tiết kiệm được nếu xóa nút này (đối với ma trận bất đối xứng)
            cost = (state.distance_matrix[prev, node] + 
                    state.distance_matrix[node, next_node] - 
                    state.distance_matrix[prev, next_node])
            costs.append((cost, node, route_idx))
    
    # Sắp xếp giảm dần theo chi phí
    costs.sort(key=lambda x: x[0], reverse=True)
    
    for i in range(min(nodes_to_remove, len(costs))):
        _, node, route_idx = costs[i]
        if node not in destroyed.unassigned:
            destroyed.unassigned.append(node)
            for r in destroyed.routes:
                if node in r:
                    r.remove(node)
                    
    destroyed.routes = [r for r in destroyed.routes if len(r) > 2]
    return destroyed