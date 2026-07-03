# File chứa các hàm tiện ích quản lý cache, vị trí node và xây dựng danh sách lân cận Granular.
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from Algorithms.Tabu.structures import Route, Solution


def build_granular_lists(
    matrix:        np.ndarray,
    n:             int,
    granular_beta: float,
    granular_k:    int,
) -> Dict[int, List[int]]:
    # Xây dựng danh sách lân cận giới hạn (granular neighborhood) cho từng khách hàng.
    customers = list(range(1, n))
    depot_dists = matrix[0, 1:].astype(float)
    avg_dist = float(np.mean(depot_dists[depot_dists > 0])) if len(depot_dists) > 0 else 1.0
    threshold = granular_beta * avg_dist * 2
    neighbors: Dict[int, List[int]] = {}
    for i in customers:
        row = matrix[i].astype(float)
        eligible = [j for j in customers if j != i and row[j] <= threshold]
        if len(eligible) < granular_k:
            k_part = min(granular_k + 2, n)
            part_indices = np.argpartition(row, k_part - 1)[:k_part]
            sorted_part = part_indices[np.argsort(row[part_indices])]
            eligible = [int(j) for j in sorted_part if j != 0 and j != i][:granular_k]
        else:
            eligible.sort(key=lambda j: row[j])
            eligible = eligible[:granular_k]
        neighbors[i] = eligible

    return neighbors


def build_caches(
    sol:          Solution,
    demands_arr:  np.ndarray,
    matrix:       np.ndarray,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    # Tính và lưu trữ tổng tải trọng và khoảng cách của từng tuyến đường.
    route_loads = {}
    route_dists = {}
    for i, r in enumerate(sol):
        route_loads[i] = float(sum(demands_arr[node] for node in r if node != 0))
        route_dists[i] = route_dist_raw(r, matrix)
    return route_loads, route_dists


def route_dist_raw(route: Route, matrix: np.ndarray) -> float:
    # Tính tổng khoảng cách thực tế của một tuyến đường trực tiếp từ ma trận.
    if len(route) <= 2:
        return 0.0
    return float(sum(matrix[route[i], route[i + 1]] for i in range(len(route) - 1)))


def total_cost_cached(route_dists: Dict[int, float]) -> float:
    # Lấy tổng khoảng cách tất cả các tuyến đường từ cache đã được tính sẵn.
    return sum(route_dists.values())


def penalized_cost_cached(
    sol:         Solution,
    route_loads: Dict[int, float],
    route_dists: Dict[int, float],
    capacity:    float,
    lam:         float,
) -> float:
    # Tính tổng chi phí có tính thêm khoản phạt vi phạm ràng buộc tải trọng.
    dist = 0.0
    penalty = 0.0
    for i, r in enumerate(sol):
        if len(r) <= 2:
            continue
        dist += route_dists[i]
        penalty += max(0.0, route_loads[i] - capacity)
    return dist + lam * penalty


def copy_sol(sol: Solution) -> Solution:
    # Sao chép toàn bộ nghiệm sang một cấu trúc mới độc lập.
    return [r[:] for r in sol]


def node_positions(sol: Solution) -> Dict[int, Tuple[int, int]]:
    # Tạo ánh xạ từ node đến cặp (chỉ số tuyến, vị trí trong tuyến).
    pos = {}
    for ri, route in enumerate(sol):
        for pi, node in enumerate(route):
            if node != 0:
                pos[node] = (ri, pi)
    return pos


def clean_empty_routes(
    sol:         Solution,
    route_loads: Dict[int, float] | None = None,
    route_dists: Dict[int, float] | None = None,
) -> None:
    # Xóa các tuyến đường rỗng và đồng bộ lại cache tải trọng và khoảng cách.
    keep_indices = [i for i, r in enumerate(sol) if len(r) > 2]
    new_sol = [sol[i] for i in keep_indices]
    sol.clear()
    sol.extend(new_sol)

    if route_loads is not None and route_dists is not None:
        new_loads = {new_i: route_loads[old_i] for new_i, old_i in enumerate(keep_indices)}
        new_dists = {new_i: route_dists[old_i] for new_i, old_i in enumerate(keep_indices)}
        route_loads.clear()
        route_loads.update(new_loads)
        route_dists.clear()
        route_dists.update(new_dists)
