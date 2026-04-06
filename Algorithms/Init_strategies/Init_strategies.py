"""
init_strategies.py
==================
Chiến lược khởi tạo nghiệm ban đầu dùng chung cho tất cả các thuật toán:
    ACO, SA, Tabu Search, ALNS, MILP.

Chỉ gồm 3 chiến lược:
    1. random_init        — xáo trộn ngẫu nhiên, nhét vào xe theo capacity
    2. greedy_init        — Nearest Neighbor Heuristic, chọn node gần nhất
    3. clarke_wright_init — Clarke-Wright Savings, tiết kiệm chi phí depot

Quy ước:
    - Node 0 là depot.
    - Ma trận khoảng cách đơn vị MÉT (int64, từ OSRM).
    - Mỗi route có dạng [0, c1, c2, ..., 0].
    - demand mỗi khách hàng mặc định = 1.
"""

from __future__ import annotations

import random
import numpy as np
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Kiểu dữ liệu
# ──────────────────────────────────────────────────────────────────────────────

Route    = List[int]    # [0, c1, c2, ..., 0]
Solution = List[Route]  # danh sách tất cả các route


# ──────────────────────────────────────────────────────────────────────────────
# Hàm tiện ích nội bộ
# ──────────────────────────────────────────────────────────────────────────────

def _build_demands(num_nodes: int,
                   demands: Optional[Dict[int, float]],
                   default_demand: float = 1.0) -> Dict[int, float]:
    """Chuẩn hóa dict demands; depot luôn = 0, khách thiếu dùng default."""
    if demands is None:
        return {i: (0.0 if i == 0 else default_demand)
                for i in range(num_nodes)}
    result = {i: demands.get(i, default_demand) for i in range(num_nodes)}
    result[0] = 0.0
    return result


def _route_load(route: Route, demands: Dict[int, float]) -> float:
    """Tính tổng demand của một route, không tính depot."""
    return sum(demands.get(n, 0.0) for n in route if n != 0)


def _route_cost(route: Route, matrix: np.ndarray) -> float:
    """Tính tổng khoảng cách (mét) của một route."""
    return float(sum(matrix[route[i], route[i + 1]]
                     for i in range(len(route) - 1)))


def _close_and_open(solution: Solution,
                    current_route: Route) -> Route:
    """Đóng route hiện tại (thêm depot), đẩy vào solution, trả về route mới."""
    current_route.append(0)
    solution.append(current_route)
    return [0]


def _merge_excess_routes(solution: Solution, max_vehicles: int) -> Solution:
    """
    Gộp route cuối vào route kề trước nếu số xe vượt max_vehicles.
    Không kiểm tra capacity khi gộp — thuật toán tối ưu tự xử lý sau.
    """
    while len(solution) > max_vehicles:
        last = solution.pop()
        solution[-1] = solution[-1][:-1] + last[1:]
    return solution


def _validate_solution(solution: Solution,
                        num_nodes: int,
                        demands: Dict[int, float],
                        capacity: float) -> Tuple[bool, List[str]]:
    """
    Kiểm tra tính hợp lệ của nghiệm:
      - Mỗi khách hàng được thăm đúng 1 lần.
      - Không vi phạm capacity.
      - Mỗi route bắt đầu và kết thúc tại depot.
    Trả về (is_valid, danh_sách_lỗi).
    """
    errors:  List[str]      = []
    visited: Dict[int, int] = {}

    for idx, route in enumerate(solution):
        if not route or route[0] != 0 or route[-1] != 0:
            errors.append(
                f"Route {idx} không bắt đầu/kết thúc tại depot: {route}")
        load = _route_load(route, demands)
        if load > capacity:
            errors.append(
                f"Route {idx} vượt capacity: {load:.0f} > {capacity:.0f}")
        for node in route:
            if node != 0:
                visited[node] = visited.get(node, 0) + 1

    missing    = set(range(1, num_nodes)) - set(visited)
    duplicates = {n: c for n, c in visited.items() if c > 1}

    if missing:
        errors.append(
            f"{len(missing)} khách hàng chưa được phục vụ: "
            f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
    if duplicates:
        errors.append(
            f"{len(duplicates)} khách hàng bị thăm nhiều lần: "
            f"{list(duplicates.items())[:5]}")

    return (len(errors) == 0), errors


# ──────────────────────────────────────────────────────────────────────────────
# Chiến lược 1: RANDOM
# ──────────────────────────────────────────────────────────────────────────────

def random_init(matrix: np.ndarray,
                num_nodes: int,
                capacity: float,
                demands: Optional[Dict[int, float]] = None,
                default_demand: float = 1.0,
                max_vehicles: int = 200,
                seed: Optional[int] = None) -> Solution:
    """
    Khởi tạo NGẪU NHIÊN.

    Xáo trộn thứ tự khách hàng, nhét lần lượt vào xe hiện tại;
    khi vượt capacity thì đóng xe và mở xe mới.

    Ưu điểm : Nhanh, tạo đa dạng nghiệm (dùng seed khác nhau).
    Nhược điểm: Chất lượng thấp, chi phí ban đầu cao.
    Dùng cho  : SA, Tabu (khám phá không gian rộng), ALNS (diversification).
    """
    rng         = random.Random(seed)
    demands_map = _build_demands(num_nodes, demands, default_demand)
    customers   = list(range(1, num_nodes))
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

    return _merge_excess_routes(solution, max_vehicles)


# ──────────────────────────────────────────────────────────────────────────────
# Chiến lược 2: GREEDY (Nearest Neighbor Heuristic)
# ──────────────────────────────────────────────────────────────────────────────

def greedy_init(matrix: np.ndarray,
                num_nodes: int,
                capacity: float,
                demands: Optional[Dict[int, float]] = None,
                default_demand: float = 1.0,
                max_vehicles: int = 200) -> Solution:
    """
    Khởi tạo THAM LAM — Nearest Neighbor Heuristic (NNH).

    Từ depot, mỗi bước chọn khách hàng chưa thăm GẦN NHẤT còn vừa
    capacity. Khi không còn ai vừa, về depot và mở xe mới.

    Ưu điểm : Chất lượng tốt hơn random, nhanh O(n²).
    Nhược điểm: Dễ rơi vào local optimum, không đa dạng.
    Dùng cho  : ACO (khởi tạo pheromone τ₀), SA, Tabu, ALNS (intensification).
    """
    demands_map   = _build_demands(num_nodes, demands, default_demand)
    unvisited     = set(range(1, num_nodes))
    solution:      Solution = []
    vehicle_count = 0

    while unvisited and vehicle_count < max_vehicles:
        current_route: Route = [0]
        current              = 0
        current_load         = 0.0
        vehicle_count       += 1

        while unvisited:
            best_node: Optional[int] = None
            best_dist                = float("inf")

            for node in unvisited:
                if current_load + demands_map[node] > capacity:
                    continue
                d = float(matrix[current, node])
                if d < best_dist:
                    best_dist = d
                    best_node = node

            if best_node is None:
                break  # Không còn node nào vừa capacity → đóng xe

            current_route.append(best_node)
            current_load += demands_map[best_node]
            unvisited.remove(best_node)
            current = best_node

        current_route.append(0)
        solution.append(current_route)

    # Fallback: khách còn lại do đạt max_vehicles
    if unvisited:
        for node in sorted(unvisited):
            solution[-1].insert(-1, node)

    return solution


# ──────────────────────────────────────────────────────────────────────────────
# Chiến lược 3: CLARKE-WRIGHT SAVINGS
# ──────────────────────────────────────────────────────────────────────────────

def clarke_wright_init(matrix: np.ndarray,
                        num_nodes: int,
                        capacity: float,
                        demands: Optional[Dict[int, float]] = None,
                        default_demand: float = 1.0,
                        max_vehicles: int = 200) -> Solution:
    """
    Khởi tạo CLARKE-WRIGHT SAVINGS.

    Tính savings s(i,j) = d(0,i) + d(j,0) - d(i,j) cho mọi cặp khách.
    Sắp xếp giảm dần. Lần lượt hợp nhất 2 route nếu:
      - i là node cuối route A, j là node đầu route B (ghép A→B).
      - Tổng load sau gộp không vượt capacity.

    Ý tưởng tiết kiệm: thay depot→i→depot + depot→j→depot (2 xe)
    bằng depot→i→j→depot (1 xe), tiết kiệm d(0,i)+d(j,0)-d(i,j).

    Ưu điểm : Ít xe nhất trong 3 chiến lược, chất lượng cao nhất.
    Nhược điểm: Chậm hơn O(n² log n), nghiệm ít đa dạng.
    Dùng cho  : ALNS (nghiệm ban đầu tốt), MILP (upper bound chặt hơn).
    """
    demands_map    = _build_demands(num_nodes, demands, default_demand)
    customers      = list(range(1, num_nodes))

    # Mỗi khách khởi đầu là 1 route độc lập
    routes:        Dict[int, Route] = {i: [0, i, 0] for i in customers}
    loads:         Dict[int, float] = {i: demands_map[i] for i in customers}
    node_to_route: Dict[int, int]   = {i: i for i in customers}

    # Tính savings s(i,j)
    savings: List[Tuple[float, int, int]] = []
    for i in customers:
        for j in customers:
            if i >= j:
                continue
            s = (float(matrix[0, i])
                 + float(matrix[j, 0])
                 - float(matrix[i, j]))
            savings.append((s, i, j))
    savings.sort(key=lambda x: -x[0])  # Giảm dần

    for s, i, j in savings:
        if s <= 0:
            break  # Không còn lợi ích khi gộp

        ri = node_to_route.get(i)
        rj = node_to_route.get(j)
        if ri is None or rj is None or ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        # i phải là cuối route_i, j phải là đầu route_j
        if route_i[-2] != i or route_j[1] != j:
            continue

        if loads[ri] + loads[rj] > capacity:
            continue

        # Hợp nhất: bỏ depot cuối route_i + depot đầu route_j
        merged     = route_i[:-1] + route_j[1:]
        routes[ri] = merged
        loads[ri] += loads[rj]
        del routes[rj]
        del loads[rj]

        for node in route_j:
            if node != 0:
                node_to_route[node] = ri

    solution = list(routes.values())
    return _merge_excess_routes(solution, max_vehicles)


# ──────────────────────────────────────────────────────────────────────────────
# Factory — entry-point thống nhất
# ──────────────────────────────────────────────────────────────────────────────

STRATEGY_MAP = {
    "random":        random_init,
    "greedy":        greedy_init,
    "clarke_wright": clarke_wright_init,
}


def get_init_strategy(name: str):
    """
    Trả về hàm khởi tạo theo tên.
    Tên hợp lệ: 'random', 'greedy', 'clarke_wright'.
    Raise ValueError nếu tên không hợp lệ.
    """
    if name not in STRATEGY_MAP:
        raise ValueError(
            f"Chiến lược '{name}' không hợp lệ. "
            f"Chọn trong: {list(STRATEGY_MAP.keys())}"
        )
    return STRATEGY_MAP[name]


def init_solution(strategy: str,
                  matrix: np.ndarray,
                  num_nodes: int,
                  capacity: float,
                  demands: Optional[Dict[int, float]] = None,
                  default_demand: float = 1.0,
                  max_vehicles: int = 9999,
                  seed: Optional[int] = None,
                  validate: bool = True) -> Solution:
    """
    Entry-point thống nhất: khởi tạo nghiệm theo chiến lược chọn.

    Parameters
    ----------
    strategy       : 'random' | 'greedy' | 'clarke_wright'
    matrix         : Ma trận khoảng cách (mét, shape [N, N])
    num_nodes      : Tổng số node (1 depot + N-1 khách hàng)
    capacity       : Sức chứa mỗi xe
    demands        : Dict {node_id: demand}; None → dùng default_demand
    default_demand : Demand mặc định khi demands=None
    max_vehicles   : Số xe tối đa
    seed           : Seed ngẫu nhiên (chỉ dùng cho 'random')
    validate       : In thông tin và cảnh báo sau khi khởi tạo

    Returns
    -------
    Solution — list các route, mỗi route dạng [0, ..., 0].

    Examples
    --------
    >>> sol = init_solution("greedy", matrix, 1600, capacity=10)
    >>> sol = init_solution("random", matrix, 1600, capacity=10, seed=42)
    >>> sol = init_solution("clarke_wright", matrix, 1600, capacity=10)
    """
    fn          = get_init_strategy(strategy)
    demands_map = _build_demands(num_nodes, demands, default_demand)

    if strategy == "random":
        solution = fn(matrix, num_nodes, capacity, demands_map,
                      default_demand, max_vehicles, seed=seed)
    else:
        solution = fn(matrix, num_nodes, capacity, demands_map,
                      default_demand, max_vehicles)

    if validate:
        is_valid, errors = _validate_solution(
            solution, num_nodes, demands_map, capacity)
        total_cost = sum(_route_cost(r, matrix) for r in solution)
        print(f"[Init:{strategy}] "
              f"{len(solution)} xe | "
              f"{total_cost:.0f}m ({total_cost / 1000:.2f}km) | "
              f"{'OK' if is_valid else 'WARN'}")
        for err in errors:
            print(f"  [WARN] {err}")

    return solution