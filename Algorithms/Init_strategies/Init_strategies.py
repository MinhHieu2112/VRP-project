# File định nghĩa các chiến lược khởi tạo nghiệm ban đầu (Random, Greedy, Clarke-Wright) cho VRP.
from __future__ import annotations

import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from Utils.Operators.local_search import merge_excess_routes_safe, validate_solution, route_cost, Route, Solution

KM_SCALE = 100

def _build_demands(num_nodes: int,
                   demands: Optional[Dict[int, float]],
                   default_demand: float = 1.0) -> Dict[int, float]:
    # Chuẩn hóa dict demands đảm bảo depot = 0 và các khách hàng thiếu dùng default.
    if demands is None:
        return {i: (0.0 if i == 0 else default_demand)
                for i in range(num_nodes)}
    result = {i: demands.get(i, default_demand) for i in range(num_nodes)}
    result[0] = 0.0
    return result


def _close_and_open(solution: Solution, current_route: Route) -> Route:
    # Đóng tuyến đường hiện tại và mở tuyến đường mới.
    current_route.append(0)
    solution.append(current_route)
    return [0]


def random_init(matrix: np.ndarray,
                num_nodes: int,
                capacity: float,
                demands_map: Dict[int, float],
                max_vehicles: int = 200,
                seed: Optional[int] = None) -> Solution:
    # Khởi tạo nghiệm ngẫu nhiên bằng cách xáo trộn thứ tự khách hàng và phân xe theo tải trọng.
    rng       = random.Random(seed)
    customers = list(range(1, num_nodes))
    rng.shuffle(customers)

    solution:      Solution = []
    current_route: Route    = [0]
    current_load            = 0.0

    for node in customers:
        d = demands_map[node]
        if current_load + d > capacity:
            current_route = _close_and_open(solution, current_route)
            current_load  = 0.0
        current_route.append(node)
        current_load += d

    if len(current_route) > 1:
        current_route.append(0)
        solution.append(current_route)

    return merge_excess_routes_safe(solution, max_vehicles, demands_map, capacity)


def greedy_init(matrix: np.ndarray,
                num_nodes: int,
                capacity: float,
                demands_map: Dict[int, float],
                max_vehicles: int = 200) -> Solution:
    # Khởi tạo nghiệm tham lam bằng heuristic láng giềng gần nhất được vector hóa.

    # Chuyển demands sang numpy array để vectorized
    demands_arr = np.array([demands_map.get(i, 0.0) for i in range(num_nodes)],
                           dtype=np.float64)

    unvisited     = np.ones(num_nodes, dtype=bool)
    unvisited[0]  = False  # Depot không phải customer

    solution:      Solution = []
    vehicle_count = 0

    while unvisited.any() and vehicle_count < max_vehicles:
        current_route: Route = [0]
        current              = 0
        current_load         = 0.0
        vehicle_count       += 1

        while True:
            # Tạo mask: còn unvisited VÀ vừa capacity VÀ không phải depot
            feasible_mask = (
                unvisited &
                (demands_arr + current_load <= capacity)
            )
            feasible_mask[0] = False  # Không chọn depot

            if not feasible_mask.any():
                break  # Không còn node vừa capacity → đóng xe

            # Lấy row khoảng cách từ current, chỉ xét feasible
            row = matrix[current].astype(np.float64)
            # Gán inf cho node không feasible
            row_masked = np.where(feasible_mask, row, np.inf)
            best_node  = int(np.argmin(row_masked))

            current_route.append(best_node)
            current_load       += demands_arr[best_node]
            unvisited[best_node] = False
            current              = best_node

        current_route.append(0)
        solution.append(current_route)

    # Fallback: khách còn lại nếu đạt max_vehicles
    if unvisited.any():
        remaining = np.where(unvisited)[0]
        for node in remaining:
            solution[-1].insert(-1, int(node))

    return merge_excess_routes_safe(solution, max_vehicles, demands_map, capacity)


def clarke_wright_init(matrix: np.ndarray,
                        num_nodes: int,
                        capacity: float,
                        demands_map: Dict[int, float],
                        max_vehicles: int = 200) -> Solution:
    # Khởi tạo nghiệm bằng thuật toán Clarke-Wright Savings hợp nhất các tuyến theo mức tiết kiệm.

    customers  = list(range(1, num_nodes))
    depot_dist = matrix[0].astype(np.float64)    # d(depot → i) = d(0, i)
    back_dist  = matrix[:, 0].astype(np.float64) # d(i → depot) = d(i, 0)

    # Mỗi khách khởi đầu là 1 route độc lập [0, i, 0]
    routes:        Dict[int, Route] = {i: [0, i, 0] for i in customers}
    loads:         Dict[int, float] = {i: float(demands_map[i]) for i in customers}
    node_to_route: Dict[int, int]   = {i: i for i in customers}

    savings: List[Tuple[float, int, int]] = []
    for i in customers:
        for j in customers:
            if i == j:
                continue  # chỉ bỏ cặp (i,i), GIỮ cả (i,j) lẫn (j,i)
            s = (float(depot_dist[i])   # d(0 → i): chi phí nếu i là cuối route
                 + float(back_dist[j])  # d(j → 0): chi phí nếu j là đầu route
                 - float(matrix[i, j])) # d(i → j): chi phí nối trực tiếp
            if s > 0:
                savings.append((s, i, j))

    savings.sort(key=lambda x: -x[0])  # Giảm dần theo savings

    for s, i, j in savings:
        ri = node_to_route.get(i)
        rj = node_to_route.get(j)
        if ri is None or rj is None or ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        if route_i[-2] != i or route_j[1] != j:
            continue
        if loads[ri] + loads[rj] > capacity:
            continue

        # Hợp nhất: cắt depot cuối route_i, cắt depot đầu route_j
        merged     = route_i[:-1] + route_j[1:]
        routes[ri] = merged
        loads[ri] += loads[rj]
        del routes[rj]
        del loads[rj]

        for node in route_j:
            if node != 0:
                node_to_route[node] = ri

    solution = list(routes.values())
    # [DRY-FIX] Dùng hàm util chung — có kiểm tra capacity khi gộp
    return merge_excess_routes_safe(solution, max_vehicles, demands_map, capacity)


STRATEGY_MAP = {
    "random":        random_init,
    "greedy":        greedy_init,
    "clarke_wright": clarke_wright_init,
}


def init_solution(strategy: str,
                  matrix: np.ndarray,
                  num_nodes: int,
                  capacity: float,
                  demands: Optional[Dict[int, float]] = None,
                  default_demand: float = 1.0,
                  max_vehicles: int = 9999,
                  seed: Optional[int] = None,
                  validate: bool = True) -> Solution:
    # Điểm vào thống nhất để khởi tạo nghiệm ban đầu theo chiến lược được chọn.
    if strategy not in STRATEGY_MAP:
        raise ValueError(
            f"Chiến lược '{strategy}' không hợp lệ. "
            f"Chọn trong: {list(STRATEGY_MAP.keys())}"
        )

    demands_map = _build_demands(num_nodes, demands, default_demand)

    init_func = STRATEGY_MAP[strategy]
    kwargs = {}
    if strategy == "random":
        kwargs["seed"] = seed
    solution = init_func(matrix, num_nodes, capacity, demands_map, max_vehicles, **kwargs)

    if validate:
        is_valid, errors = validate_solution(
            solution, num_nodes, demands_map, capacity)
        total_units = sum(route_cost(matrix, r) for r in solution)
        total_km    = total_units / KM_SCALE

        print(f"[Init:{strategy}] "
              f"{len(solution)} xe | "
              f"{total_units:.0f} units ({total_km:.2f} km) | "
              f"{'OK' if is_valid else 'WARN'}")
        for err in errors:
            print(f"  [WARN] {err}")

    return solution