# Wrapper mỏng tái sử dụng các hàm tính chi phí từ Utils.local_search thay vì triển khai lại.
from __future__ import annotations
import numpy as np
from typing import Dict, List

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
