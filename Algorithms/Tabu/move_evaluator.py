# File chịu trách nhiệm tính toán delta chi phí ước lượng cho các dịch chuyển của GTS.
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from Algorithms.Tabu.structures import Solution

def eval_relocate(
    sol:         Solution,
    route_loads: Dict[int, float],
    u:           int,
    r_src:       int,
    p_u:         int,
    r_dst:       int,
    p_ins:       int,
    matrix:      np.ndarray,
    demands_arr: np.ndarray,
    capacity:    float,
    lam:         float,
    remove_gain: float,
) -> float:
    # Tính toán sự thay đổi chi phí khi di chuyển một khách hàng sang vị trí mới dựa trên remove_gain tính trước.
    route_d = sol[r_dst]

    prev_d = route_d[p_ins - 1]
    next_d = route_d[p_ins]
    insert_cost = matrix[prev_d, u] + matrix[u, next_d] - matrix[prev_d, next_d]
    delta = float(insert_cost - remove_gain)

    u_dem = demands_arr[u]
    load_s = route_loads[r_src]
    load_d = route_loads[r_dst]
    delta += lam * (
        max(0.0, load_s - u_dem - capacity)
        + max(0.0, load_d + u_dem - capacity)
        - max(0.0, load_s - capacity)
        - max(0.0, load_d - capacity)
    )
    return delta


def eval_relocate2(
    sol:         Solution,
    route_loads: Dict[int, float],
    u:           int,
    v:           int,
    r_src:       int,
    p_u:         int,
    r_dst:       int,
    p_ins:       int,
    matrix:      np.ndarray,
    demands_arr: np.ndarray,
    capacity:    float,
    lam:         float,
    remove_gain: float,
) -> Optional[float]:
    # Tính toán sự thay đổi chi phí khi di chuyển hai khách hàng liên tiếp sang vị trí mới dựa trên remove_gain tính trước.
    route_d = sol[r_dst]

    prev_d = route_d[p_ins - 1]
    next_d = route_d[p_ins]
    insert_cost = (
        matrix[prev_d, u] + matrix[u, v] + matrix[v, next_d]
        - matrix[prev_d, next_d]
    )
    delta = float(insert_cost - remove_gain)

    seg_dem = demands_arr[u] + demands_arr[v]
    load_s = route_loads[r_src]
    load_d = route_loads[r_dst]
    delta += lam * (
        max(0.0, load_s - seg_dem - capacity)
        + max(0.0, load_d + seg_dem - capacity)
        - max(0.0, load_s - capacity)
        - max(0.0, load_d - capacity)
    )
    return delta


def eval_swap(
    sol:         Solution,
    route_loads: Dict[int, float],
    u:           int,
    v:           int,
    r_u:         int,
    p_u:         int,
    r_v:         int,
    p_v:         int,
    matrix:      np.ndarray,
    demands_arr: np.ndarray,
    capacity:    float,
    lam:         float,
) -> Optional[float]:
    # Tính toán sự thay đổi chi phí khi đổi chỗ hai khách hàng.
    route_u = sol[r_u]
    route_v = sol[r_v]

    pu_prev = route_u[p_u - 1]
    pu_next = route_u[p_u + 1]
    pv_prev = route_v[p_v - 1]
    pv_next = route_v[p_v + 1]

    if r_u == r_v:
        if abs(p_u - p_v) == 1:
            return None
        old = matrix[pu_prev, u] + matrix[u, pu_next] + matrix[pv_prev, v] + matrix[v, pv_next]
        new = matrix[pu_prev, v] + matrix[v, pu_next] + matrix[pv_prev, u] + matrix[u, pv_next]
        return float(new - old)
    else:
        old = matrix[pu_prev, u] + matrix[u, pu_next] + matrix[pv_prev, v] + matrix[v, pv_next]
        new = matrix[pu_prev, v] + matrix[v, pu_next] + matrix[pv_prev, u] + matrix[u, pv_next]
        delta = float(new - old)

        u_dem = demands_arr[u]
        v_dem = demands_arr[v]
        load_u = route_loads[r_u]
        load_v = route_loads[r_v]
        delta += lam * (
            max(0.0, load_u - u_dem + v_dem - capacity)
            + max(0.0, load_v - v_dem + u_dem - capacity)
            - max(0.0, load_u - capacity)
            - max(0.0, load_v - capacity)
        )
        return delta


def eval_2opt_star(
    sol:            Solution,
    route_loads:    Dict[int, float],
    r1:             int,
    i:              int,
    r2:             int,
    j:              int,
    matrix:         np.ndarray,
    suffix_demands: Dict[int, list[float]],
    capacity:       float,
    lam:            float,
    avg_edge:       float,
) -> Optional[float]:
    # Tính toán sự thay đổi chi phí khi thực hiện phép lai chéo 2-opt* cắt nối đuôi tuyến sử dụng suffix_demands tối ưu O(1).
    route1 = sol[r1]
    route2 = sol[r2]

    if i == 0 or i >= len(route1) - 1:
        return None
    if j == 0 or j >= len(route2) - 1:
        return None
    if len(route1) < 4 or len(route2) < 4:
        return None

    A = route1[i]
    C = route1[i + 1]
    B = route2[j]
    D = route2[j + 1]
    delta = float(matrix[A, D] + matrix[B, C] - matrix[A, C] - matrix[B, D])

    if delta > avg_edge * 2:
        return None

    tail1_load = suffix_demands[r1][i + 1]
    tail2_load = suffix_demands[r2][j + 1]
    head1_load = route_loads[r1] - tail1_load
    head2_load = route_loads[r2] - tail2_load

    delta += lam * (
        max(0.0, head1_load + tail2_load - capacity)
        + max(0.0, head2_load + tail1_load - capacity)
        - max(0.0, route_loads[r1] - capacity)
        - max(0.0, route_loads[r2] - capacity)
    )
    return delta
