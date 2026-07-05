# Thư viện tập trung hóa các toán tử local search và tiện ích xử lý nghiệm dùng chung cho mọi thuật toán VRP.
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np

Route    = List[int]
Solution = List[Route]



# Nhóm 1: Tính chi phí và tải trọng cơ bản


def route_cost(matrix: np.ndarray, route: Route) -> float:
    # Tính tổng khoảng cách thực tế của một tuyến đường theo đúng thứ tự chiều đi (tương thích bất đối xứng).
    if len(route) < 2:
        return 0.0
    return float(sum(matrix[route[i], route[i + 1]] for i in range(len(route) - 1)))


def route_load(demands: Dict[int, float], route: Route) -> float:
    # Tính tổng tải trọng của các khách hàng trong một tuyến đường.
    return sum(demands.get(n, 0.0) for n in route if n != 0)



# Nhóm 2: Or-opt nội tuyến (tương thích ma trận bất đối xứng)


def or_opt_intra(matrix: np.ndarray, route: Route, max_passes: int = 50) -> Route:
    # Tối ưu hóa nội tuyến Or-opt-1: rút một node và chèn lại vào vị trí tốt nhất mà không đảo chiều cung.
    if len(route) <= 3:
        return route[:]

    best = route[:]
    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        n = len(best)

        for i in range(1, n - 1):
            node   = best[i]
            prev_i = best[i - 1]
            next_i = best[i + 1]

            gain_remove = (
                matrix[prev_i, node] + matrix[node, next_i] - matrix[prev_i, next_i]
            )

            best_gain = 1e-6
            best_j    = -1

            for j in range(1, n - 1):
                if j == i or j == i - 1:
                    continue
                prev_j = best[j - 1]
                next_j = best[j]
                gain_insert = (
                    matrix[prev_j, node] + matrix[node, next_j] - matrix[prev_j, next_j]
                )
                gain = gain_remove - gain_insert
                if gain > best_gain:
                    best_gain = gain
                    best_j    = j

            if best_j != -1:
                tmp = best[:]
                tmp.pop(i)
                insert_at = best_j if best_j < i else best_j - 1
                tmp.insert(insert_at, node)
                best     = tmp
                improved = True
                break

    return best


# Nhóm 3: Cập nhật gia tăng nghiệm (Tabu-style cache dict)

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
    # Áp dụng phép chuyển đổi 1 node sang vị trí mới và cập nhật cache gia tăng.
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
    # Áp dụng phép chuyển đổi 2 node liên tiếp sang vị trí mới và cập nhật cache gia tăng.
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
    # Áp dụng phép đổi chỗ hai node giữa hai tuyến đường và cập nhật cache gia tăng.
    route_u = sol[r_u]
    route_v = sol[r_v]

    pu_prev = route_u[p_u - 1]
    pu_next = route_u[p_u + 1]
    pv_prev = route_v[p_v - 1]
    pv_next = route_v[p_v + 1]

    sol[r_u][p_u] = v
    sol[r_v][p_v] = u

    if r_u == r_v:
        dist_delta = (
            matrix[pu_prev, v] + matrix[v, pu_next]
            + matrix[pv_prev, u] + matrix[u, pv_next]
            - matrix[pu_prev, u] - matrix[u, pu_next]
            - matrix[pv_prev, v] - matrix[v, pv_next]
        )
        route_dists[r_u] += dist_delta
    else:
        du_delta = (
            matrix[pu_prev, v] + matrix[v, pu_next]
            - matrix[pu_prev, u] - matrix[u, pu_next]
        )
        dv_delta = (
            matrix[pv_prev, u] + matrix[u, pv_next]
            - matrix[pv_prev, v] - matrix[v, pv_next]
        )
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
    # Áp dụng phép lai chéo 2-opt* cắt và ghép đuôi hai tuyến đường không đảo chiều cung.
    new_r1 = sol[r1][:i + 1] + sol[r2][j + 1:]
    new_r2 = sol[r2][:j + 1] + sol[r1][i + 1:]
    sol[r1] = new_r1
    sol[r2] = new_r2

    route_loads[r1] = float(sum(demands_arr[node] for node in new_r1 if node != 0))
    route_loads[r2] = float(sum(demands_arr[node] for node in new_r2 if node != 0))
    route_dists[r1] = route_cost(matrix, new_r1)
    route_dists[r2] = route_cost(matrix, new_r2)



# Nhóm 4: Granular candidate list


def build_granular_lists(
    matrix:        np.ndarray,
    n:             int,
    granular_beta: float,
    granular_k:    int,
) -> Dict[int, List[int]]:
    # Xây dựng danh sách lân cận giới hạn (granular neighborhood) để tăng tốc tìm kiếm lân cận.
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
            sorted_part  = part_indices[np.argsort(row[part_indices])]
            eligible = [int(j) for j in sorted_part if j != 0 and j != i][:granular_k]
        else:
            eligible.sort(key=lambda j: row[j])
            eligible = eligible[:granular_k]
        neighbors[i] = eligible

    return neighbors



# Nhóm 5: Tiện ích xử lý nghiệm (validate, gộp tuyến)


def _find_best_receiver(
    solution: Solution,
    last_load: float,
    demands: Dict[int, float],
    capacity: float,
) -> int:
    # Tìm kiếm tuyến đường tốt nhất trong nghiệm hiện tại để gộp một tuyến đường có tải trọng last_load (ưu tiên feasible, fallback là tải thấp nhất).
    best_idx: int = -1
    best_load: float = float("inf")
    feasible_found = False

    for i, route in enumerate(solution):
        r_load = route_load(demands, route)
        would_exceed = (r_load + last_load) > capacity

        if not would_exceed:
            if not feasible_found or r_load < best_load:
                best_idx = i
                best_load = r_load
                feasible_found = True
        elif not feasible_found and r_load < best_load:
            best_idx = i
            best_load = r_load

    return best_idx


def _merge_into(
    solution: Solution,
    receiver_idx: int,
    last_route: Route,
    last_load: float,
    capacity: float,
    demands: Dict[int, float],
) -> None:
    # Thực hiện gộp tuyến đường last_route vào tuyến đường tại receiver_idx và in cảnh báo nếu vi phạm tải trọng.
    receiver = solution[receiver_idx]
    recv_load = route_load(demands, receiver)
    merged_load = recv_load + last_load

    if merged_load > capacity:
        print(
            f"[local_search][WARN] Gộp route bắt buộc vào route[{receiver_idx}] "
            f"vi phạm capacity: {merged_load:.0f} > {capacity:.0f}. "
            f"Solution tạm thời infeasible — penalty sẽ xử lý sau."
        )

    solution[receiver_idx] = receiver[:-1] + last_route[1:]


def merge_excess_routes_safe(
    solution:     Solution,
    max_vehicles: int,
    demands:      Dict[int, float],
    capacity:     float,
) -> Solution:
    # Điều phối gộp các tuyến xe dư thừa sao cho tối thiểu hóa việc vượt tải trọng của xe.
    while len(solution) > max_vehicles:
        last = solution.pop()
        last_load = route_load(demands, last)

        best_idx = _find_best_receiver(solution, last_load, demands, capacity)

        if best_idx == -1:
            warnings.warn(
                "[local_search] merge_excess_routes_safe: solution rỗng sau khi pop, "
                "không thể gộp. Route bị bỏ qua.",
                stacklevel=2,
            )
            break

        _merge_into(solution, best_idx, last, last_load, capacity, demands)

    solution[:] = [r for r in solution if len(r) > 2]
    return solution


def validate_solution(
    solution:    Solution,
    num_nodes:   int,
    demands:     Dict[int, float],
    capacity:    float,
    raise_on_error: bool = False,
) -> tuple[bool, list[str]]:
    # Kiểm tra tính hợp lệ toàn diện của nghiệm (depot, tải trọng, trùng lặp và thiếu sót khách hàng).
    errors:  list[str]      = []
    visited: dict[int, int] = {}

    for idx, route in enumerate(solution):
        if not route or route[0] != 0 or route[-1] != 0:
            errors.append(
                f"Route {idx} không bắt đầu/kết thúc tại depot: {route}"
            )
        load = route_load(demands, route)
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
