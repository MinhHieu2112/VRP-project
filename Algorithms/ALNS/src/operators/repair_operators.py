import numpy as np


def get_actual_vehicle_count(routes):
    return len([r for r in routes if len(r) > 2])


def _best_insertions_for_node(node, node_demand, routes, route_loads, dist, capacity, top_k=2):
    """
    Tìm top_k vị trí chèn tốt nhất cho một node.
    Trả về list (cost, r_idx, pos_idx) đã sort tăng dần, tối đa top_k phần tử.
    Dùng partial sort thay vì sort toàn bộ để tiết kiệm thời gian.
    """
    best = []  # list (cost, r_idx, pos_idx), giữ tối đa top_k

    for r_idx, route in enumerate(routes):
        if route_loads[r_idx] + node_demand > capacity:
            continue
        route_len = len(route)
        for i in range(1, route_len):
            prev, nxt = route[i - 1], route[i]
            cost = dist[prev, node] + dist[node, nxt] - dist[prev, nxt]

            if len(best) < top_k:
                best.append((cost, r_idx, i))
                if len(best) == top_k:
                    best.sort(key=lambda x: x[0])
            elif cost < best[-1][0]:
                best[-1] = (cost, r_idx, i)
                best.sort(key=lambda x: x[0])

    return best


def greedy_insertion(state, random_state):
    """Chèn tham lam: vị trí có chi phí tăng thêm thấp nhất."""
    repaired = state.copy()
    random_state.shuffle(repaired.unassigned)

    max_v = state.config['constraints'].get('max_vehicles', 200)
    dist = repaired.distance_matrix
    nodes_to_process = list(repaired.unassigned)
    repaired.unassigned = []

    for node in nodes_to_process:
        node_demand = repaired.demands[node]
        best = _best_insertions_for_node(
            node, node_demand, repaired.routes,
            repaired.route_loads, dist, repaired.capacity, top_k=1
        )

        if best:
            cost, r_idx, pos_idx = best[0]
            repaired.routes[r_idx].insert(pos_idx, node)
            repaired.route_loads[r_idx] += node_demand
        else:
            if get_actual_vehicle_count(repaired.routes) < max_v:
                repaired.routes.append([0, node, 0])
                repaired.route_loads.append(node_demand)
            else:
                # Fallback: chèn vào route ít tải nhất
                # BUG FIX: kiểm tra routes và route_loads không rỗng,
                # và đảm bảo min_idx nằm trong phạm vi routes hợp lệ.
                valid = [(i, repaired.route_loads[i])
                         for i in range(len(repaired.routes))
                         if len(repaired.routes[i]) > 0]
                if valid:
                    min_idx = min(valid, key=lambda x: x[1])[0]
                    repaired.routes[min_idx].insert(-1, node)
                    repaired.route_loads[min_idx] += node_demand
                else:
                    repaired.routes.append([0, node, 0])
                    repaired.route_loads.append(node_demand)

    return repaired


def regret_insertion(state, random_state):
    """
    Chèn hối tiếc (Regret-2): ưu tiên node khó chèn nhất.

    Tối ưu so với bản gốc:
    - Dùng _best_insertions_for_node (partial sort, top-2 only)
    - Cache insertion costs, chỉ recompute node vừa bị ảnh hưởng
      (route vừa thay đổi) thay vì recompute toàn bộ mỗi vòng.
    """
    repaired = state.copy()
    max_v = state.config['constraints'].get('max_vehicles', 200)
    dist = repaired.distance_matrix

    unassigned = list(repaired.unassigned)
    repaired.unassigned = []

    # Tính cache lần đầu cho tất cả nodes
    # cache[node] = (regret_value, best_insertion or None)
    def compute_cache(node):
        node_demand = repaired.demands[node]
        top2 = _best_insertions_for_node(
            node, node_demand, repaired.routes,
            repaired.route_loads, dist, repaired.capacity, top_k=2
        )
        if len(top2) >= 2:
            regret = top2[1][0] - top2[0][0]
            return (regret, top2[0])
        elif len(top2) == 1:
            return (1e6 + top2[0][0], top2[0])
        else:
            return (-1.0, None)

    cache = {node: compute_cache(node) for node in unassigned}

    dirty = set()  # nodes cần recompute cache

    while unassigned:
        # Chọn node có regret lớn nhất
        best_node = max(unassigned, key=lambda n: cache[n][0])
        _, best_insertion = cache[best_node]

        unassigned.remove(best_node)
        del cache[best_node]

        if best_insertion is not None:
            cost, r_idx, pos_idx = best_insertion
            repaired.routes[r_idx].insert(pos_idx, best_node)
            repaired.route_loads[r_idx] += repaired.demands[best_node]
            # Chỉ những node có best_insertion trỏ vào route r_idx mới cần recompute
            dirty = {
                n for n in unassigned
                if cache[n][1] is not None and cache[n][1][1] == r_idx
            }
            # Thêm những node chưa tìm được chỗ (có thể route mới tạo ra slot)
            dirty |= {n for n in unassigned if cache[n][1] is None}
        else:
            if get_actual_vehicle_count(repaired.routes) < max_v:
                repaired.routes.append([0, best_node, 0])
                repaired.route_loads.append(repaired.demands[best_node])
                # Route mới → tất cả node chưa gán đều có thể chèn vào
                dirty = set(unassigned)
            else:
                # Fallback: chèn vào route ít tải nhất
                # BUG FIX: same guard as greedy_insertion
                valid = [(i, repaired.route_loads[i])
                         for i in range(len(repaired.routes))
                         if len(repaired.routes[i]) > 0]
                if valid:
                    min_idx = min(valid, key=lambda x: x[1])[0]
                    repaired.routes[min_idx].insert(-1, best_node)
                    repaired.route_loads[min_idx] += repaired.demands[best_node]
                else:
                    repaired.routes.append([0, best_node, 0])
                    repaired.route_loads.append(repaired.demands[best_node])
                dirty = set(unassigned)

        # Recompute chỉ những node bị ảnh hưởng
        for n in dirty:
            cache[n] = compute_cache(n)

    return repaired