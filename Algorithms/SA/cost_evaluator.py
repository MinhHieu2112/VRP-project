# Wrapper mỏng tái sử dụng các hàm tính chi phí từ Utils.local_search thay vì triển khai lại.
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

from Utils.Operators.local_search import route_cost, route_load

Route    = List[int]
Solution = List[Route]


def total_cost(
    dist:            np.ndarray,
    solution:        Solution,
    vehicle_penalty: int = 3000,
) -> float:
    # Tính tổng chi phí khoảng cách và chi phí phạt phương tiện của toàn bộ nghiệm.
    cost = 0.0
    for r in solution:
        if len(r) > 2:
            cost += route_cost(dist, r) + vehicle_penalty
    return cost


def build_route_costs(dist: np.ndarray, solution: Solution) -> List[float]:
    # Xây dựng danh sách chi phí khoảng cách cho tất cả các tuyến đường.
    return [route_cost(dist, r) for r in solution]


def is_feasible_load(
    demands_map: Dict[int, float],
    route:       Route,
    capacity:    float,
) -> bool:
    # Kiểm tra xem tuyến đường có thỏa mãn giới hạn tải trọng hay không.
    return route_load(demands_map, route) <= capacity


def eval_swap_delta(
    dist: np.ndarray,
    r1: Route,
    r2: Route,
    idx1: int,
    idx2: int,
) -> Tuple[float, float]:
    # Tính delta chi phí khoảng cách ở O(1) khi tráo đổi hai khách hàng giữa hai tuyến đường khác nhau (bất đối xứng).
    u, v = r1[idx1], r2[idx2]
    prev_u, next_u = r1[idx1 - 1], r1[idx1 + 1]
    prev_v, next_v = r2[idx2 - 1], r2[idx2 + 1]

    delta_r1 = (dist[prev_u, v] + dist[v, next_u]) - (dist[prev_u, u] + dist[u, next_u])
    delta_r2 = (dist[prev_v, u] + dist[u, next_v]) - (dist[prev_v, v] + dist[v, next_v])

    return float(delta_r1), float(delta_r2)


def eval_relocate_delta(
    dist: np.ndarray,
    r1: Route,
    r2: Route,
    idx1: int,
    ins_pos: int,
) -> Tuple[float, float]:
    # Tính delta chi phí khoảng cách ở O(1) khi chuyển một khách hàng từ tuyến này sang tuyến khác (bất đối xứng).
    u = r1[idx1]
    prev_u, next_u = r1[idx1 - 1], r1[idx1 + 1]
    prev_v, next_v = r2[ins_pos - 1], r2[ins_pos]

    delta_r1 = dist[prev_u, next_u] - (dist[prev_u, u] + dist[u, next_u])
    delta_r2 = (dist[prev_v, u] + dist[u, next_v]) - dist[prev_v, next_v]

    return float(delta_r1), float(delta_r2)


def eval_intra_swap_delta(
    dist: np.ndarray,
    r1: Route,
    i: int,
    j: int,
) -> float:
    # Tính delta chi phí khoảng cách ở O(1) khi tráo đổi hai khách hàng trong cùng một tuyến đường (bất đối xứng).
    if i == j:
        return 0.0
    if i > j:
        i, j = j, i

    u, v = r1[i], r1[j]
    prev_u = r1[i - 1]
    next_v = r1[j + 1]

    if j == i + 1:
        old_cost = dist[prev_u, u] + dist[u, v] + dist[v, next_v]
        new_cost = dist[prev_u, v] + dist[v, u] + dist[u, next_v]
        return float(new_cost - old_cost)
    else:
        next_u = r1[i + 1]
        prev_v = r1[j - 1]
        old_cost = dist[prev_u, u] + dist[u, next_u] + dist[prev_v, v] + dist[v, next_v]
        new_cost = dist[prev_u, v] + dist[v, next_u] + dist[prev_v, u] + dist[u, next_v]
        return float(new_cost - old_cost)
