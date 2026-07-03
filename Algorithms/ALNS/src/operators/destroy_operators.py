# File định nghĩa các toán tử phá hủy (destroy operators) để loại bỏ các khách hàng khỏi lộ trình trong ALNS.
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
    # Lọc bỏ các tuyến đường rỗng và đồng bộ danh sách tải trọng tương ứng.
    kept = [(r, load) for r, load in zip(state.routes, state.route_loads) if len(r) > 2]
    if kept:
        state.routes, state.route_loads = map(list, zip(*kept))
    else:
        state.routes = []
        state.route_loads = []


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
                break

    _cleanup_empty_routes(destroyed)
    return destroyed


def worst_removal(state: "CvrpState", rng: rnd.Generator, **kwargs) -> "CvrpState":
    # Toán tử loại bỏ các khách hàng có chi phí tăng thêm cao nhất trong lộ trình.
    destroyed = state.copy()
    nodes_to_remove = get_nodes_to_remove(state)

    removed_count = 0
    removed_nodes = set()

    while removed_count < nodes_to_remove:
        costs = []
        for route_idx, route in enumerate(destroyed.routes):
            for i in range(1, len(route) - 1):
                node = route[i]
                if node in removed_nodes:
                    continue
                prev, next_node = route[i-1], route[i+1]
                cost = (destroyed.distance_matrix[prev, node] +
                        destroyed.distance_matrix[node, next_node] -
                        destroyed.distance_matrix[prev, next_node])
                costs.append((cost, node, route_idx))

        if not costs:
            break

        costs.sort(key=lambda x: x[0], reverse=True)
        _, node, _ = costs[0]

        removed_nodes.add(node)
        destroyed.unassigned.append(node)
        for r_idx, r in enumerate(destroyed.routes):
            if node in r:
                r.remove(node)
                destroyed.route_loads[r_idx] -= destroyed.demands[node]
                break
        removed_count += 1

    _cleanup_empty_routes(destroyed)
    return destroyed