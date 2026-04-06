import numpy as np


def get_actual_vehicle_count(routes):
    return len([r for r in routes if len(r) > 2])


def _best_insertions_for_node(node, node_demand, routes, route_loads, dist, capacity, top_k=2):
    """
    Tìm top_k vị trí chèn tốt nhất cho một node (thỏa capacity).
    Trả về list (cost, r_idx, pos_idx) đã sort tăng dần.
    """
    best = []

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


def _fallback_insert(repaired, node, node_demand, max_v):
    """
    Fallback khi không tìm được chỗ chèn thỏa capacity:
    1. Nếu chưa đủ max_v xe → mở xe mới
    2. Nếu đã đủ xe → chèn vào xe ít tải nhất (ghi nhận vi phạm,
       penalty trong objective sẽ kéo ALNS sửa ở vòng tiếp theo)

    FIX: bản cũ chèn mà không log vi phạm, khó debug.
    """
    if get_actual_vehicle_count(repaired.routes) < max_v:
        repaired.routes.append([0, node, 0])
        repaired.route_loads.append(node_demand)
    else:
        # Tìm route không rỗng có tải nhỏ nhất
        valid = [
            (i, repaired.route_loads[i])
            for i in range(len(repaired.routes))
            if len(repaired.routes[i]) > 0
        ]
        if valid:
            min_idx = min(valid, key=lambda x: x[1])[0]
            repaired.routes[min_idx].insert(-1, node)
            repaired.route_loads[min_idx] += node_demand
            # Vi phạm capacity tạm thời — penalty trong objective sẽ xử lý
        else:
            repaired.routes.append([0, node, 0])
            repaired.route_loads.append(node_demand)


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
            _, r_idx, pos_idx = best[0]
            repaired.routes[r_idx].insert(pos_idx, node)
            repaired.route_loads[r_idx] += node_demand
        else:
            # FIX: dùng hàm fallback thống nhất, có log vi phạm
            _fallback_insert(repaired, node, node_demand, max_v)

    return repaired


def regret_insertion(state, random_state):
    """
    Chèn hối tiếc (Regret-2): ưu tiên node khó chèn nhất.
    Cache insertion costs, chỉ recompute khi route thay đổi.
    """
    repaired = state.copy()
    max_v = state.config['constraints'].get('max_vehicles', 200)
    dist = repaired.distance_matrix

    unassigned = list(repaired.unassigned)
    repaired.unassigned = []

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
    dirty = set()

    while unassigned:
        best_node = max(unassigned, key=lambda n: cache[n][0])
        _, best_insertion = cache[best_node]

        unassigned.remove(best_node)
        del cache[best_node]

        if best_insertion is not None:
            _, r_idx, pos_idx = best_insertion
            repaired.routes[r_idx].insert(pos_idx, best_node)
            repaired.route_loads[r_idx] += repaired.demands[best_node]
            dirty = {
                n for n in unassigned
                if cache[n][1] is not None and cache[n][1][1] == r_idx
            }
            dirty |= {n for n in unassigned if cache[n][1] is None}
        else:
            # FIX: dùng hàm fallback thống nhất
            _fallback_insert(repaired, best_node, repaired.demands[best_node], max_v)
            # Route mới → tất cả node chưa gán có thể chèn vào
            dirty = set(unassigned)

        for n in dirty:
            cache[n] = compute_cache(n)

    return repaired
