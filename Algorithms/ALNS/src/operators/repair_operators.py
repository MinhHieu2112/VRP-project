# Định nghĩa các toán tử tái thiết (repair operators) để chèn lại các khách hàng chưa gán vào lộ trình.
import numpy as np
import numpy.random as rnd
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.state import CvrpState


def get_actual_vehicle_count(routes: list) -> int:
    # Đếm số lượng xe thực tế đang thực hiện lộ trình (độ dài tuyến đường lớn hơn 2).
    return sum(1 for r in routes if len(r) > 2)


def _best_insertions_for_node(
    node,
    node_demand,
    routes,
    route_loads,
    dist,
    capacity,
    granular_neighbors,
    pos_map,
    top_k=2,
) -> list:
    # Tìm kiếm các vị trí chèn tốt nhất cho khách hàng thỏa mãn ràng buộc tải trọng dựa trên granular lân cận.
    best = []
    candidates = set()

    for v in granular_neighbors.get(node, []):
        if v in pos_map:
            r_idx, pos_idx = pos_map[v]
            candidates.add((r_idx, pos_idx))
            candidates.add((r_idx, pos_idx + 1))

    for r_idx in range(len(routes)):
        r_len = len(routes[r_idx])
        if r_len > 2:
            candidates.add((r_idx, 1))
            candidates.add((r_idx, r_len - 1))

    for r_idx, i in candidates:
        if i < 1 or i >= len(routes[r_idx]):
            continue
        if route_loads[r_idx] + node_demand > capacity:
            continue

        route = routes[r_idx]
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
    # Cơ chế chèn dự phòng khi không tìm được vị trí chèn hợp lệ nào thỏa mãn tải trọng.
    if get_actual_vehicle_count(repaired.routes) < max_v:
        route = [0, node, 0]
        repaired.routes.append(route)
        repaired.route_loads.append(node_demand)
        repaired.route_costs.append(repaired.route_cost(route))
    else:
        valid = [(i, repaired.route_loads[i])
                 for i in range(len(repaired.routes))
                 if len(repaired.routes[i]) > 0]
        if valid:
            min_idx = min(valid, key=lambda x: x[1])[0]
            repaired.routes[min_idx].insert(-1, node)
            repaired.route_loads[min_idx] += node_demand
            repaired.route_costs[min_idx] = repaired.route_cost(repaired.routes[min_idx])
        else:
            route = [0, node, 0]
            repaired.routes.append(route)
            repaired.route_loads.append(node_demand)
            repaired.route_costs.append(repaired.route_cost(route))


def greedy_insertion(state: "CvrpState", rng: rnd.Generator, **kwargs) -> "CvrpState":
    # Tái thiết lời giải bằng phương pháp chèn tham lam lần lượt các khách hàng chưa gán dùng granular.
    repaired = state
    rng.shuffle(repaired.unassigned)

    constraints = repaired.config.get(
        'global_constraints', repaired.config.get('constraints', {}))
    max_v = constraints.get('max_vehicles', 200)
    dist  = repaired.distance_matrix
    granular_neighbors = repaired.config.get("granular_neighbors", {})

    pos_map = {}
    for r_idx, route in enumerate(repaired.routes):
        for pos_idx, node in enumerate(route):
            if node != 0:
                pos_map[node] = (r_idx, pos_idx)

    nodes_to_process  = list(repaired.unassigned)
    repaired.unassigned = []

    for node in nodes_to_process:
        node_demand = repaired.demands[node]
        best = _best_insertions_for_node(
            node, node_demand,
            repaired.routes, repaired.route_loads,
            dist, repaired.capacity,
            granular_neighbors, pos_map, top_k=1,
        )
        if best:
            _, r_idx, pos_idx = best[0]
            repaired.routes[r_idx].insert(pos_idx, node)
            repaired.route_loads[r_idx] += node_demand
            repaired.route_costs[r_idx] = repaired.route_cost(repaired.routes[r_idx])

            route = repaired.routes[r_idx]
            for idx in range(pos_idx, len(route)):
                v = route[idx]
                if v != 0:
                    pos_map[v] = (r_idx, idx)
        else:
            _fallback_insert(repaired, node, node_demand, max_v)
            for r_idx, route in enumerate(repaired.routes):
                if node in route:
                    for pos_idx, n in enumerate(route):
                        if n != 0:
                            pos_map[n] = (r_idx, pos_idx)
                    break

    return repaired


def regret_insertion(state: "CvrpState", rng: rnd.Generator, **kwargs) -> "CvrpState":
    # Tái thiết lời giải dựa trên độ hối tiếc Regret-2 khi chèn khách hàng sử dụng granular.
    repaired = state
    constraints = repaired.config.get(
        'global_constraints', repaired.config.get('constraints', {}))
    max_v = constraints.get('max_vehicles', 200)
    dist  = repaired.distance_matrix
    granular_neighbors = repaired.config.get("granular_neighbors", {})

    pos_map = {}
    for r_idx, route in enumerate(repaired.routes):
        for pos_idx, node in enumerate(route):
            if node != 0:
                pos_map[node] = (r_idx, pos_idx)

    unassigned         = list(repaired.unassigned)
    repaired.unassigned = []

    def compute_entry(node):
        # Tính toán chi phí hối tiếc của việc không chèn node vào vị trí tốt nhất.
        node_demand = repaired.demands[node]
        top2 = _best_insertions_for_node(
            node, node_demand,
            repaired.routes, repaired.route_loads,
            dist, repaired.capacity,
            granular_neighbors, pos_map, top_k=2,
        )
        if len(top2) >= 2:
            return (top2[1][0] - top2[0][0], top2[0])
        elif len(top2) == 1:
            return (1e9, top2[0])
        else:
            return (-1.0, None)

    cache = {node: compute_entry(node) for node in unassigned}

    while unassigned:
        best_node = max(unassigned, key=lambda n: cache[n][0])
        _, best_insertion = cache[best_node]

        unassigned.remove(best_node)
        del cache[best_node]

        if best_insertion is not None:
            _, r_idx, pos_idx = best_insertion
            repaired.routes[r_idx].insert(pos_idx, best_node)
            repaired.route_loads[r_idx] += repaired.demands[best_node]
            repaired.route_costs[r_idx] = repaired.route_cost(repaired.routes[r_idx])

            route = repaired.routes[r_idx]
            for idx in range(pos_idx, len(route)):
                v = route[idx]
                if v != 0:
                    pos_map[v] = (r_idx, idx)
        else:
            _fallback_insert(repaired, best_node, repaired.demands[best_node], max_v)
            for r_idx, route in enumerate(repaired.routes):
                if best_node in route:
                    for pos_idx, n in enumerate(route):
                        if n != 0:
                            pos_map[n] = (r_idx, pos_idx)
                    break

        for n in unassigned:
            cache[n] = compute_entry(n)

    return repaired