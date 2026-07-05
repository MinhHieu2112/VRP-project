# File chứa các hàm tiện ích quản lý cache, vị trí node và danh sách lân cận Granular dành riêng cho Tabu.
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from Algorithms.Tabu.structures import Route, Solution
from Utils.Operators.local_search import route_cost as _route_cost


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
    # Tính tổng khoảng cách thực tế của một tuyến đường — wrapper gọi Utils.local_search.route_cost.
    return _route_cost(matrix, route)


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
