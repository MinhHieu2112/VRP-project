# File chứa các hàm tính toán khoảng cách, tải trọng và ràng buộc của các tuyến đường.
from __future__ import annotations
import numpy as np
from typing import Dict, List

Route    = List[int]
Solution = List[Route]

def route_cost(dist: np.ndarray, route: Route) -> float:
    # Tính tổng khoảng cách thực tế của một tuyến đường.
    return float(
        sum(dist[route[i], route[i + 1]] for i in range(len(route) - 1))
    )


def route_load(demands_map: Dict[int, float], route: Route) -> float:
    # Tính tổng lượng hàng khách hàng yêu cầu trên một tuyến đường.
    return sum(demands_map.get(n, 0.0) for n in route if n != 0)


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
