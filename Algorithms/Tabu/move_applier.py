# File chịu trách nhiệm áp dụng các phép biến đổi (relocate, swap, 2-opt*) lên nghiệm và cập nhật cache.
from __future__ import annotations
from typing import Dict
import numpy as np
from Algorithms.Tabu.structures import Solution
from Algorithms.Tabu.utils import route_dist_raw

def apply_relocate(
    sol:         Solution,
    route_loads: Dict[int, float],
    route_dists: Dict[int, float],
    u:           int,
    r_src:       int,
    p_u:         int,
    r_dst:       int,
    p_ins:       int,
    matrix:      np.ndarray,
    demands_arr: np.ndarray,
) -> None:
    # Áp dụng phép chuyển đổi 1 node sang vị trí mới và cập nhật cache.
    route_s = sol[r_src]
    route_d = sol[r_dst]

    prev_u = route_s[p_u - 1]
    next_u = route_s[p_u + 1]
    prev_d = route_d[p_ins - 1]
    next_d = route_d[p_ins]

    dist_del = matrix[prev_u, u] + matrix[u, next_u] - matrix[prev_u, next_u]
    dist_ins = matrix[prev_d, u] + matrix[u, next_d] - matrix[prev_d, next_d]

    route_s.pop(p_u)
    route_d.insert(p_ins, u)

    u_dem = demands_arr[u]
    route_loads[r_src] -= u_dem
    route_loads[r_dst] += u_dem
    route_dists[r_src] -= dist_del
    route_dists[r_dst] += dist_ins


def apply_relocate2(
    sol:         Solution,
    route_loads: Dict[int, float],
    route_dists: Dict[int, float],
    u:           int,
    v:           int,
    r_src:       int,
    p_u:         int,
    r_dst:       int,
    p_ins:       int,
    matrix:      np.ndarray,
    demands_arr: np.ndarray,
) -> None:
    # Áp dụng phép chuyển đổi 2 node liên tiếp sang vị trí mới và cập nhật cache.
    route_s = sol[r_src]
    route_d = sol[r_dst]

    if p_u + 2 >= len(route_s):
        return
    next_v = route_s[p_u + 2]
    prev_u = route_s[p_u - 1]
    prev_d = route_d[p_ins - 1]
    next_d = route_d[p_ins]

    dist_del = (
        matrix[prev_u, u] + matrix[u, v] + matrix[v, next_v]
        - matrix[prev_u, next_v]
    )
    dist_ins = (
        matrix[prev_d, u] + matrix[u, v] + matrix[v, next_d]
        - matrix[prev_d, next_d]
    )

    route_s.pop(p_u + 1)
    route_s.pop(p_u)
    route_d.insert(p_ins, v)
    route_d.insert(p_ins, u)

    seg_dem = demands_arr[u] + demands_arr[v]
    route_loads[r_src] -= seg_dem
    route_loads[r_dst] += seg_dem
    route_dists[r_src] -= dist_del
    route_dists[r_dst] += dist_ins


def apply_swap(
    sol:         Solution,
    route_loads: Dict[int, float],
    route_dists: Dict[int, float],
    u:           int,
    v:           int,
    r_u:         int,
    p_u:         int,
    r_v:         int,
    p_v:         int,
    matrix:      np.ndarray,
    demands_arr: np.ndarray,
) -> None:
    # Áp dụng phép đổi chỗ hai node giữa hai tuyến đường và cập nhật cache.
    route_u = sol[r_u]
    route_v = sol[r_v]

    pu_prev = route_u[p_u - 1]
    pu_next = route_u[p_u + 1]
    pv_prev = route_v[p_v - 1]
    pv_next = route_v[p_v + 1]

    dist_delta = (
        matrix[pu_prev, v] + matrix[v, pu_next] + matrix[pv_prev, u] + matrix[u, pv_next]
        - matrix[pu_prev, u] - matrix[u, pu_next] - matrix[pv_prev, v] - matrix[v, pv_next]
    )

    sol[r_u][p_u] = v
    sol[r_v][p_v] = u

    if r_u == r_v:
        route_dists[r_u] += dist_delta
    else:
        du_delta = matrix[pu_prev, v] + matrix[v, pu_next] - matrix[pu_prev, u] - matrix[u, pu_next]
        dv_delta = matrix[pv_prev, u] + matrix[u, pv_next] - matrix[pv_prev, v] - matrix[v, pv_next]
        route_dists[r_u] += du_delta
        route_dists[r_v] += dv_delta

        u_dem = demands_arr[u]
        v_dem = demands_arr[v]
        route_loads[r_u] += v_dem - u_dem
        route_loads[r_v] += u_dem - v_dem


def apply_2opt_star(
    sol:         Solution,
    route_loads: Dict[int, float],
    route_dists: Dict[int, float],
    r1:          int,
    i:           int,
    r2:          int,
    j:           int,
    matrix:      np.ndarray,
    demands_arr: np.ndarray,
) -> None:
    # Áp dụng phép lai chéo 2-opt* cắt và ghép đuôi hai tuyến đường.
    new_r1 = sol[r1][:i + 1] + sol[r2][j + 1:]
    new_r2 = sol[r2][:j + 1] + sol[r1][i + 1:]
    sol[r1] = new_r1
    sol[r2] = new_r2

    route_loads[r1] = float(sum(demands_arr[node] for node in new_r1 if node != 0))
    route_loads[r2] = float(sum(demands_arr[node] for node in new_r2 if node != 0))
    route_dists[r1] = route_dist_raw(new_r1, matrix)
    route_dists[r2] = route_dist_raw(new_r2, matrix)
