from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np

# Kiểu dữ liệu (mirror của Init_strategies để tránh import vòng)
Route    = List[int]
Solution = List[Route]

def merge_excess_routes_safe(
    solution:     Solution,
    max_vehicles: int,
    demands:      Dict[int, float],
    capacity:     float,
) -> Solution:
    
    def _route_load(route: Route) -> float:
        return sum(demands.get(n, 0.0) for n in route if n != 0)

    while len(solution) > max_vehicles:
        last = solution.pop()         
        last_load = _route_load(last)
        best_idx: int = -1
        best_load: float = float("inf")
        feasible_found = False

        for i, route in enumerate(solution):
            r_load = _route_load(route)
            would_exceed = (r_load + last_load) > capacity

            if not would_exceed:
                if not feasible_found or r_load < best_load:
                    best_idx = i
                    best_load = r_load
                    feasible_found = True
            elif not feasible_found and r_load < best_load:
                best_idx = i
                best_load = r_load

        if best_idx == -1:
            warnings.warn(
                "[vrp_utils] merge_excess_routes_safe: solution rỗng sau khi pop, "
                "không thể gộp. Route bị bỏ qua.",
                stacklevel=2,
            )
            break

        receiver = solution[best_idx]
        recv_load = _route_load(receiver)
        merged_load = recv_load + last_load

        if merged_load > capacity:
            print(
                f"[vrp_utils][WARN] Gộp route bắt buộc vào route[{best_idx}] "
                f"vi phạm capacity: {merged_load:.0f} > {capacity:.0f}. "
                f"Solution tạm thời infeasible — penalty sẽ xử lý sau."
            )

        solution[best_idx] = receiver[:-1] + last[1:]

    solution[:] = [r for r in solution if len(r) > 2]

    return solution


# ──────────────────────────────────────────────────────────────────────────────
# Hàm validate solution (nâng cấp từ Init_strategies._validate_solution)
# ──────────────────────────────────────────────────────────────────────────────

def validate_solution(
    solution:    Solution,
    num_nodes:   int,
    demands:     Dict[int, float],
    capacity:    float,
    raise_on_error: bool = False,
) -> tuple[bool, list[str]]:
    
    errors:  list[str]      = []
    visited: dict[int, int] = {}

    for idx, route in enumerate(solution):
        if not route or route[0] != 0 or route[-1] != 0:
            errors.append(
                f"Route {idx} không bắt đầu/kết thúc tại depot: {route}"
            )
        load = sum(demands.get(n, 0.0) for n in route if n != 0)
        if load > capacity:
            errors.append(
                f"Route {idx} vượt capacity: {load:.0f} > {capacity:.0f}"
            )
        for node in route:
            if node != 0:
                visited[node] = visited.get(node, 0) + 1

    missing    = set(range(1, num_nodes)) - set(visited)
    duplicates = {n: c for n, c in visited.items() if c > 1}
    if missing:
        errors.append(
            f"{len(missing)} khách chưa được phục vụ: "
            f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
        )
    if duplicates:
        errors.append(
            f"{len(duplicates)} khách bị thăm nhiều lần: "
            f"{list(duplicates.items())[:5]}"
        )

    is_valid = len(errors) == 0
    if not is_valid and raise_on_error:
        raise ValueError(
            "Solution không hợp lệ:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    return is_valid, errors
