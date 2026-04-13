"""
Algorithms/ALNS/src/operators/repair_operators.py

FIXES so với bản cũ:
  [FIX-CACHE] regret_insertion: invalidate toàn bộ cache sau mỗi lần chèn.
              Bản cũ chỉ invalidate node có best_insertion trỏ vào r_idx vừa thay đổi
              → miss những node mà route r_idx giờ rẻ hơn nhưng chưa được check.
              Đơn giản và đúng hơn: dirty = set(unassigned) sau mỗi insertion.
"""

import numpy as np


def get_actual_vehicle_count(routes: list) -> int:
    return sum(1 for r in routes if len(r) > 2)


def _best_insertions_for_node(node, node_demand, routes, route_loads,
                               dist, capacity, top_k=2) -> list:
    """
    Tìm top_k vị trí chèn tốt nhất cho node (thỏa capacity).
    Trả về list [(cost, r_idx, pos_idx)] đã sort tăng dần.
    """
    best = []
    for r_idx, route in enumerate(routes):
        if route_loads[r_idx] + node_demand > capacity:
            continue
        for i in range(1, len(route)):
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


def _fallback_insert(repaired, node: int, node_demand: int, max_v: int):
    """
    Chèn node khi không có vị trí nào thỏa capacity:
      - Còn slot xe → mở xe mới [0, node, 0]
      - Hết xe      → chèn vào xe ít tải nhất (vi phạm tạm thời,
                       penalty trong objective() sẽ dẫn ALNS sửa sau)
    """
    if get_actual_vehicle_count(repaired.routes) < max_v:
        repaired.routes.append([0, node, 0])
        repaired.route_loads.append(node_demand)
    else:
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


def greedy_insertion(state, random_state):
    """
    Chèn tham lam: với mỗi node unassigned (thứ tự ngẫu nhiên),
    chọn vị trí có chi phí tăng thêm thấp nhất trong các xe hiện có.
    """
    repaired = state.copy()
    random_state.shuffle(repaired.unassigned)

    constraints = repaired.config.get(
        'global_constraints', repaired.config.get('constraints', {}))
    max_v = constraints.get('max_vehicles', 200)
    dist  = repaired.distance_matrix

    nodes_to_process  = list(repaired.unassigned)
    repaired.unassigned = []

    for node in nodes_to_process:
        node_demand = repaired.demands[node]
        best = _best_insertions_for_node(
            node, node_demand,
            repaired.routes, repaired.route_loads,
            dist, repaired.capacity, top_k=1,
        )
        if best:
            _, r_idx, pos_idx = best[0]
            repaired.routes[r_idx].insert(pos_idx, node)
            repaired.route_loads[r_idx] += node_demand
        else:
            _fallback_insert(repaired, node, node_demand, max_v)

    return repaired


def regret_insertion(state, random_state):
    """
    Chèn hối tiếc (Regret-2):
      1. Tính regret = (2nd-best cost) - (best cost) cho mỗi node
      2. Chèn node có regret cao nhất (node khó chèn nhất)
      3. Sau mỗi lần chèn, recompute toàn bộ cache

    [FIX-CACHE] dirty = set(unassigned) — recompute tất cả thay vì
    chỉ subset dựa trên r_idx. Chi phí tính toán tăng nhẹ nhưng
    tránh chọn sai thứ tự do cache stale.
    """
    repaired = state.copy()
    constraints = repaired.config.get(
        'global_constraints', repaired.config.get('constraints', {}))
    max_v = constraints.get('max_vehicles', 200)
    dist  = repaired.distance_matrix

    unassigned         = list(repaired.unassigned)
    repaired.unassigned = []

    def compute_entry(node):
        """Trả về (regret_score, best_insertion_tuple_or_None)."""
        node_demand = repaired.demands[node]
        top2 = _best_insertions_for_node(
            node, node_demand,
            repaired.routes, repaired.route_loads,
            dist, repaired.capacity, top_k=2,
        )
        if len(top2) >= 2:
            return (top2[1][0] - top2[0][0], top2[0])
        elif len(top2) == 1:
            # Chỉ 1 vị trí → regret rất cao (ưu tiên chèn trước)
            return (1e9, top2[0])
        else:
            # Không có vị trí nào thỏa capacity → regret âm (chèn sau cùng)
            return (-1.0, None)

    # Build cache lần đầu
    cache = {node: compute_entry(node) for node in unassigned}

    while unassigned:
        # Chọn node có regret lớn nhất
        best_node = max(unassigned, key=lambda n: cache[n][0])
        _, best_insertion = cache[best_node]

        unassigned.remove(best_node)
        del cache[best_node]

        if best_insertion is not None:
            _, r_idx, pos_idx = best_insertion
            repaired.routes[r_idx].insert(pos_idx, best_node)
            repaired.route_loads[r_idx] += repaired.demands[best_node]
        else:
            _fallback_insert(
                repaired, best_node, repaired.demands[best_node], max_v)

        # [FIX-CACHE] Recompute toàn bộ cache sau mỗi lần chèn/fallback
        for n in unassigned:
            cache[n] = compute_entry(n)

    return repaired