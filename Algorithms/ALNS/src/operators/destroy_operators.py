# Định nghĩa các toán tử phá hủy (destroy operators) để loại bỏ các khách hàng khỏi lộ trình trong ALNS.
import random
import numpy as np
import numpy.random as rnd
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.state import CvrpState

def get_nodes_to_remove(state):
    # Xác định số lượng node khách hàng cần loại bỏ dựa trên tổng số khách hiện có.
    num_clients_in_routes = sum(
        len(r) - 2 for r in state.routes if len(r) > 2
    )
    n_remove = int(num_clients_in_routes * random.uniform(0.05, 0.10))
    return max(1, min(n_remove, 30))


def _cleanup_empty_routes(state):
    # Lọc bỏ các tuyến đường rỗng và đồng bộ danh sách tải trọng cùng chi phí tương ứng.
    kept = [(r, load, cost) for r, load, cost in zip(state.routes, state.route_loads, state.route_costs) if len(r) > 2]
    if kept:
        state.routes, state.route_loads, state.route_costs = map(list, zip(*kept))
    else:
        state.routes = []
        state.route_loads = []
        state.route_costs = []


def random_removal(state: "CvrpState", rng: rnd.Generator, **kwargs) -> "CvrpState":
    # Toán tử loại bỏ ngẫu nhiên một số lượng khách hàng khỏi các tuyến đường hiện tại.
    destroyed = state.copy()
    nodes_to_remove = get_nodes_to_remove(state)

    all_clients = []
    for route in destroyed.routes:
        all_clients.extend([node for node in route if node != 0])

    if not all_clients:
        return destroyed

    nodes_to_remove = min(nodes_to_remove, len(all_clients))
    to_remove = rng.choice(all_clients, nodes_to_remove, replace=False)

    for node in to_remove:
        destroyed.unassigned.append(node)
        for r_idx, route in enumerate(destroyed.routes):
            if node in route:
                route.remove(node)
                destroyed.route_loads[r_idx] -= destroyed.demands[node]
                destroyed.route_costs[r_idx] = destroyed.route_cost(route)
                break

    _cleanup_empty_routes(destroyed)
    return destroyed


def worst_removal(state: "CvrpState", rng: rnd.Generator, **kwargs) -> "CvrpState":
    # Toán tử loại bỏ các khách hàng có chi phí tăng thêm cao nhất trong lộ trình một cách tối ưu.
    destroyed = state.copy()
    nodes_to_remove = get_nodes_to_remove(state)

    node_costs = {}
    node_route_pos = {}

    for r_idx, route in enumerate(destroyed.routes):
        for i in range(1, len(route) - 1):
            node = route[i]
            prev, nxt = route[i - 1], route[i + 1]
            cost = (destroyed.distance_matrix[prev, node] +
                    destroyed.distance_matrix[node, nxt] -
                    destroyed.distance_matrix[prev, nxt])
            node_costs[node] = cost
            node_route_pos[node] = (r_idx, i)

    removed_count = 0
    while removed_count < nodes_to_remove:
        if not node_costs:
            break

        node_to_rm = max(node_costs, key=lambda k: node_costs[k])
        r_idx, pos_idx = node_route_pos[node_to_rm]

        route = destroyed.routes[r_idx]
        route.pop(pos_idx)
        destroyed.route_loads[r_idx] -= destroyed.demands[node_to_rm]
        destroyed.route_costs[r_idx] = destroyed.route_cost(route)
        destroyed.unassigned.append(node_to_rm)

        del node_costs[node_to_rm]
        del node_route_pos[node_to_rm]

        for idx in range(pos_idx, len(route) - 1):
            v = route[idx]
            if v != 0:
                node_route_pos[v] = (r_idx, idx)

        if pos_idx < len(route) - 1:
            node_curr = route[pos_idx]
            if node_curr != 0:
                prev, nxt = route[pos_idx - 1], route[pos_idx + 1]
                node_costs[node_curr] = (destroyed.distance_matrix[prev, node_curr] +
                                         destroyed.distance_matrix[node_curr, nxt] -
                                         destroyed.distance_matrix[prev, nxt])

        if pos_idx - 1 > 0:
            node_prev = route[pos_idx - 1]
            if node_prev != 0:
                prev_prev, nxt_nxt = route[pos_idx - 2], route[pos_idx]
                node_costs[node_prev] = (destroyed.distance_matrix[prev_prev, node_prev] +
                                         destroyed.distance_matrix[node_prev, nxt_nxt] -
                                         destroyed.distance_matrix[prev_prev, nxt_nxt])

        removed_count += 1

    _cleanup_empty_routes(destroyed)
    return destroyed