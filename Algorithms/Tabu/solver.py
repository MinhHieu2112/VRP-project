# Lớp điều phối chính thực thi giải thuật tối ưu hóa Granular Tabu Search (GTS).
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from Utils.Operators.local_search import merge_excess_routes_safe, or_opt_intra
from Algorithms.Tabu.structures import (
    AnyMove,
    Move2OptStar,
    MoveRel1,
    MoveRel2,
    MoveSwap,
    Route,
    Solution,
)
from Algorithms.Tabu.utils import (
    build_caches,
    build_granular_lists,
    clean_empty_routes,
    copy_sol,
    node_positions,
    penalized_cost_cached,
    route_dist_raw,
    total_cost_cached,
)
from Algorithms.Tabu.move_evaluator import (
    eval_2opt_star,
    eval_relocate,
    eval_relocate2,
    eval_swap,
)
from Algorithms.Tabu.move_applier import (
    apply_2opt_star,
    apply_relocate,
    apply_relocate2,
    apply_swap,
)

class GranularTabuSearch:
    """Bộ giải tối ưu bài toán VRP sử dụng giải thuật Granular Tabu Search (GTS)."""

    def __init__(
        self,
        distance_matrix: np.ndarray,
        demands:         Dict[int, float],
        capacity:        float,
        max_v:           int,
        tabu_size:       int = 20,
        max_iter:        int = 3000,
        max_no_improve:  int = 400,
        granular_beta:   float = 1.5,
        granular_k:      int = 15,
        penalty_lambda:  float = None,
        penalty_h:       int = 20,
    ):
        # Khởi tạo các tham số GTS, ma trận khoảng cách, các ràng buộc và candidate list.
        self.matrix = distance_matrix
        self.n = distance_matrix.shape[0]
        self.demands = demands
        self.capacity = capacity
        self.max_v = max_v

        self.tabu_size = tabu_size
        self.max_iter = max_iter
        self.max_no_improve = max_no_improve
        self.granular_beta = granular_beta
        self.granular_k = granular_k
        self.pen_h = penalty_h

        nonzero = distance_matrix[distance_matrix > 0]
        avg_edge = float(np.mean(nonzero)) if len(nonzero) > 0 else 100.0
        if penalty_lambda is None:
            self.lam = avg_edge / max(capacity, 1.0)
        else:
            auto_lam = avg_edge / max(capacity, 1.0)
            self.lam = max(penalty_lambda, auto_lam * 0.5)

        self._avg_edge = avg_edge
        self._early_exit_threshold = -self._avg_edge * 0.15

        self.tau_min = max(5, tabu_size // 2)
        self.tau_max = max(15, tabu_size * 2)

        self._granular_neighbors = build_granular_lists(
            self.matrix, self.n, self.granular_beta, self.granular_k
        )

        self._demands_arr = np.array(
            [demands.get(i, 0.0) for i in range(self.n)], dtype=np.float64
        )

        avg_edges = sum(len(v) for v in self._granular_neighbors.values()) / max(
            len(self._granular_neighbors), 1
        )
        print(
            f"[GTS] Khởi tạo GTS: β={self.granular_beta}, "
            f"avg_nb={avg_edges:.1f}/node, λ={self.lam:.1f}, "
            f"early_exit_thresh={self._early_exit_threshold:.1f}"
        )

    def _double_bridge(self, sol: Solution) -> Solution:
        # Thực hiện đột phá nghiệm (shake) bằng phép biến đổi Double-Bridge.
        new_sol = copy_sol(sol)
        k = min(8, len(new_sol) // 2)
        if k < 2:
            return new_sol

        idx_to_shake = random.sample(range(len(new_sol)), k)
        flat = []
        for idx in sorted(idx_to_shake, reverse=True):
            route = new_sol.pop(idx)
            flat.extend([node for node in route if node != 0])

        if len(flat) < 4:
            return copy_sol(sol)

        positions = sorted(random.sample(range(1, len(flat)), min(4, len(flat) - 1)))
        while len(positions) < 4:
            positions.append(len(flat))
        a, b, c, _ = positions[:4]

        seg0 = flat[:a]
        seg1 = flat[a:b]
        seg2 = flat[b:c]
        seg3 = flat[c:]
        new_flat = seg0 + seg2 + seg1 + seg3

        shaken = self._rebuild_from_flat(new_flat)
        new_sol.extend(shaken)
        return new_sol

    def _rebuild_from_flat(self, flat: List[int]) -> Solution:
        # Dựng lại phương án hợp lệ từ danh sách phẳng khách hàng.
        sol: Solution = []
        curr: Route = [0]
        load = 0.0
        for node in flat:
            dem = self._demands_arr[node]
            if load + dem > self.capacity:
                curr.append(0)
                sol.append(curr)
                curr = [0]
                load = 0.0
            curr.append(node)
            load += dem
        curr.append(0)
        sol.append(curr)
        return merge_excess_routes_safe(sol, self.max_v, self.demands, self.capacity)

    def _intra_or_opt(self, sol: Solution) -> Solution:
        # Tối ưu hóa cục bộ nội tuyến Or-opt-1 bằng cách gọi Utils.local_search.or_opt_intra.
        for idx, r in enumerate(sol):
            sol[idx] = or_opt_intra(self.matrix, r)
        return sol

    def _explore_relocate1(
        self,
        u: int,
        r_u: int,
        p_u: int,
        remove_gain: float,
        pos_map: dict,
        curr_sol: list,
        route_loads: dict,
        base_cost: float,
        best_dist: float,
        iteration: int,
        tabu_dict: dict,
        best_move_delta: float,
    ) -> tuple[float, Optional[MoveRel1], Optional[tuple], bool]:
        # Tìm kiếm dịch chuyển Relocate 1 node tốt nhất cho nút u.
        local_best_delta = best_move_delta
        local_best_move = None
        local_best_key = None
        found_early_exit = False

        for v in self._granular_neighbors.get(u, []):
            if v not in pos_map:
                continue
            r_v, p_v = pos_map[v]
            if r_v == r_u:
                continue

            for p_ins in (p_v, p_v + 1):
                if p_ins < 1 or p_ins >= len(curr_sol[r_v]):
                    continue
                delta = eval_relocate(
                    curr_sol,
                    route_loads,
                    u,
                    r_u,
                    p_u,
                    r_v,
                    p_ins,
                    self.matrix,
                    self._demands_arr,
                    self.capacity,
                    self.lam,
                    remove_gain,
                )
                key = ("R1", u, r_v)
                in_tabu = tabu_dict.get(key, -1) >= iteration
                asp = base_cost + delta < best_dist - 1e-6
                if (not in_tabu or asp) and delta < local_best_delta:
                    local_best_delta = delta
                    local_best_move = MoveRel1(u, r_u, p_u, r_v, p_ins)
                    local_best_key = key
                    if delta < self._early_exit_threshold:
                        found_early_exit = True
                        break
            if found_early_exit:
                break

        return local_best_delta, local_best_move, local_best_key, found_early_exit

    def _explore_relocate2(
        self,
        u: int,
        v_next: int,
        r_u: int,
        p_u: int,
        remove_gain: float,
        pos_map: dict,
        curr_sol: list,
        route_loads: dict,
        base_cost: float,
        best_dist: float,
        iteration: int,
        tabu_dict: dict,
        best_move_delta: float,
    ) -> tuple[float, Optional[MoveRel2], Optional[tuple]]:
        # Tìm kiếm dịch chuyển Relocate 2 nodes tốt nhất cho nút u và v_next.
        local_best_delta = best_move_delta
        local_best_move = None
        local_best_key = None

        for nb in self._granular_neighbors.get(u, [])[:8]:
            if nb not in pos_map:
                continue
            r_nb, p_nb = pos_map[nb]
            if r_nb == r_u:
                continue
            for p_ins in range(1, min(len(curr_sol[r_nb]), 6)):
                delta2 = eval_relocate2(
                    curr_sol,
                    route_loads,
                    u,
                    v_next,
                    r_u,
                    p_u,
                    r_nb,
                    p_ins,
                    self.matrix,
                    self._demands_arr,
                    self.capacity,
                    self.lam,
                    remove_gain,
                )
                if delta2 is None:
                    continue
                key2 = ("R2", u, r_nb)
                in_t2 = tabu_dict.get(key2, -1) >= iteration
                asp2 = base_cost + delta2 < best_dist - 1e-6
                if (not in_t2 or asp2) and delta2 < local_best_delta:
                    local_best_delta = delta2
                    local_best_move = MoveRel2(u, v_next, r_u, p_u, r_nb, p_ins)
                    local_best_key = key2
            break

        return local_best_delta, local_best_move, local_best_key

    def _explore_swap(
        self,
        u: int,
        r_u: int,
        p_u: int,
        pos_map: dict,
        curr_sol: list,
        route_loads: dict,
        base_cost: float,
        best_dist: float,
        iteration: int,
        tabu_dict: dict,
        best_move_delta: float,
    ) -> tuple[float, Optional[MoveSwap], Optional[tuple], bool]:
        # Tìm kiếm dịch chuyển Swap tốt nhất cho nút u.
        local_best_delta = best_move_delta
        local_best_move = None
        local_best_key = None
        found_early_exit = False

        for v in self._granular_neighbors.get(u, []):
            if v not in pos_map:
                continue
            r_v, p_v = pos_map[v]
            if r_u == r_v and p_u >= p_v:
                continue
            ds = eval_swap(
                curr_sol,
                route_loads,
                u,
                v,
                r_u,
                p_u,
                r_v,
                p_v,
                self.matrix,
                self._demands_arr,
                self.capacity,
                self.lam,
            )
            if ds is None:
                continue
            keys = ("SW", min(u, v), max(u, v))
            in_t_s = tabu_dict.get(keys, -1) >= iteration
            asp_s = base_cost + ds < best_dist - 1e-6
            if (not in_t_s or asp_s) and ds < local_best_delta:
                local_best_delta = ds
                local_best_move = MoveSwap(u, v, r_u, p_u, r_v, p_v)
                local_best_key = keys
                if ds < self._early_exit_threshold:
                    found_early_exit = True
                    break

        return local_best_delta, local_best_move, local_best_key, found_early_exit

    def _compute_route_suffix_demands(self, route: Route) -> List[float]:
        # Tính suffix sum cho tuyến đường
        n_r = len(route)
        suffixes = [0.0] * n_r
        val = 0.0
        for idx in range(n_r - 1, -1, -1):
            node = route[idx]
            if node != 0:
                val += self._demands_arr[node]
            suffixes[idx] = val
        return suffixes

    def _build_suffix_demands(self, sol: Solution) -> Dict[int, List[float]]:
        # Xây dựng mảng suffix demands cho toàn bộ nghiệm
        suffix_demands = {}
        for r_idx, r in enumerate(sol):
            suffix_demands[r_idx] = self._compute_route_suffix_demands(r)
        return suffix_demands

    def _update_pos_map_for_routes(self, pos_map: dict, curr_sol: list, route_indices: list):
        # Cập nhật gia tăng pos_map cho các route được chỉ định
        for r_idx in route_indices:
            for p_idx, node in enumerate(curr_sol[r_idx]):
                if node != 0:
                    pos_map[node] = (r_idx, p_idx)

    def _explore_2optstar(
        self,
        u: int,
        r_u: int,
        p_u: int,
        pos_map: dict,
        curr_sol: list,
        route_loads: dict,
        base_cost: float,
        best_dist: float,
        iteration: int,
        tabu_dict: dict,
        best_move_delta: float,
        suffix_demands: dict,
    ) -> tuple[float, Optional[Move2OptStar], Optional[tuple]]:
        # Tìm kiếm dịch chuyển 2-opt* tốt nhất cho nút u sử dụng suffix_demands O(1).
        local_best_delta = best_move_delta
        local_best_move = None
        local_best_key = None

        for v in self._granular_neighbors.get(u, [])[:6]:
            if v not in pos_map:
                continue
            r_v, p_v = pos_map[v]
            if r_v == r_u:
                continue
            d2s = eval_2opt_star(
                curr_sol,
                route_loads,
                r_u,
                p_u,
                r_v,
                p_v,
                self.matrix,
                suffix_demands,
                self.capacity,
                self.lam,
                self._avg_edge,
            )
            if d2s is None:
                continue
            key2s = ("2S", r_u, p_u, r_v, p_v)
            in_t_2s = tabu_dict.get(key2s, -1) >= iteration
            asp_2s = base_cost + d2s < best_dist - 1e-6
            if (not in_t_2s or asp_2s) and d2s < local_best_delta:
                local_best_delta = d2s
                local_best_move = Move2OptStar(r_u, p_u, r_v, p_v)
                local_best_key = key2s

        return local_best_delta, local_best_move, local_best_key

    def solve(self, initial_solution: Solution) -> Tuple[Solution, float]:
        # Thực hiện vòng lặp GTS tìm kiếm lời giải tối ưu.
        curr_sol = copy_sol(initial_solution)
        clean_empty_routes(curr_sol)

        route_loads, route_dists = build_caches(curr_sol, self._demands_arr, self.matrix)

        best_sol = copy_sol(curr_sol)
        best_dist = total_cost_cached(route_dists)

        tabu_dict: Dict[tuple, int] = {}
        no_improve = 0
        iteration = 0
        infeasible_count = 0

        # Khởi tạo pos_map và suffix_demands một lần trước vòng lặp chính
        pos_map = node_positions(curr_sol)
        suffix_demands = self._build_suffix_demands(curr_sol)

        print(
            f"[GTS] Bắt đầu: {len(curr_sol)} xe | {best_dist / 100:.2f} km | λ={self.lam:.1f} | "
            f"max_iter={self.max_iter} | max_no_improve={self.max_no_improve}"
        )

        while iteration < self.max_iter:
            if no_improve >= self.max_no_improve:
                print(f"[GTS] Dừng iter {iteration}: {no_improve} vòng không cải thiện")
                break

            if no_improve == self.max_no_improve // 2:
                perturbed = self._double_bridge(copy_sol(best_sol))
                clean_empty_routes(perturbed)
                p_loads, p_dists = build_caches(perturbed, self._demands_arr, self.matrix)
                p_cost = total_cost_cached(p_dists)
                if p_cost < best_dist:
                    best_dist = p_cost
                    best_sol = copy_sol(perturbed)
                    no_improve = 0
                    print(f"  [GTS] Perturbation NEW BEST: {best_dist / 100:.2f} km")
                if p_cost < best_dist * 1.1:
                    curr_sol = perturbed
                    route_loads = p_loads
                    route_dists = p_dists
                    # Rebuild pos_map và suffix_demands cho perturbed solution mới
                    pos_map = node_positions(curr_sol)
                    suffix_demands = self._build_suffix_demands(curr_sol)

            base_cost = penalized_cost_cached(
                curr_sol, route_loads, route_dists, self.capacity, self.lam
            )

            best_move_delta = float("inf")
            best_move: Optional[AnyMove] = None
            best_move_key = None
            best_move_type = None
            found_early_exit = False

            for u in list(pos_map.keys()):
                if found_early_exit:
                    break

                r_u, p_u = pos_map[u]
                route_u = curr_sol[r_u]

                prev_u = route_u[p_u - 1]
                next_u = route_u[p_u + 1]
                remove_gain_rel1 = self.matrix[prev_u, u] + self.matrix[u, next_u] - self.matrix[prev_u, next_u]

                # 1. Relocate 1
                best_move_delta, m_rel1, k_rel1, found_early_exit = self._explore_relocate1(
                    u, r_u, p_u, remove_gain_rel1, pos_map, curr_sol, route_loads,
                    base_cost, best_dist, iteration, tabu_dict, best_move_delta
                )
                if m_rel1 is not None:
                    best_move = m_rel1
                    best_move_key = k_rel1
                    best_move_type = "rel1"

                if found_early_exit:
                    break

                # 2. Relocate 2
                if p_u + 1 < len(route_u) - 1:
                    v_next = route_u[p_u + 1]
                    if v_next != 0:
                        next_v = route_u[p_u + 2]
                        remove_gain_rel2 = (
                            self.matrix[prev_u, u]
                            + self.matrix[u, v_next]
                            + self.matrix[v_next, next_v]
                            - self.matrix[prev_u, next_v]
                        )
                        best_move_delta, m_rel2, k_rel2 = self._explore_relocate2(
                            u, v_next, r_u, p_u, remove_gain_rel2, pos_map, curr_sol, route_loads,
                            base_cost, best_dist, iteration, tabu_dict, best_move_delta
                        )
                        if m_rel2 is not None:
                            best_move = m_rel2
                            best_move_key = k_rel2
                            best_move_type = "rel2"

                # 3. Swap
                best_move_delta, m_swap, k_swap, found_early_exit = self._explore_swap(
                    u, r_u, p_u, pos_map, curr_sol, route_loads,
                    base_cost, best_dist, iteration, tabu_dict, best_move_delta
                )
                if m_swap is not None:
                    best_move = m_swap
                    best_move_key = k_swap
                    best_move_type = "swap"

                if found_early_exit:
                    break

                # 4. 2-opt*
                best_move_delta, m_2opts, k_2opts = self._explore_2optstar(
                    u, r_u, p_u, pos_map, curr_sol, route_loads,
                    base_cost, best_dist, iteration, tabu_dict, best_move_delta, suffix_demands
                )
                if m_2opts is not None:
                    best_move = m_2opts
                    best_move_key = k_2opts
                    best_move_type = "2opts"

            if best_move is None:
                no_improve += 1
                iteration += 1
                continue

            affected_routes = []
            if best_move_type == "rel1":
                assert isinstance(best_move, MoveRel1)
                apply_relocate(
                    curr_sol,
                    route_loads,
                    route_dists,
                    best_move.u,
                    best_move.r_src,
                    best_move.p_u,
                    best_move.r_dst,
                    best_move.p_ins,
                    self.matrix,
                    self._demands_arr,
                )
                affected_routes = [best_move.r_src, best_move.r_dst]
            elif best_move_type == "rel2":
                assert isinstance(best_move, MoveRel2)
                apply_relocate2(
                    curr_sol,
                    route_loads,
                    route_dists,
                    best_move.u,
                    best_move.v_nxt,
                    best_move.r_src,
                    best_move.p_u,
                    best_move.r_dst,
                    best_move.p_ins,
                    self.matrix,
                    self._demands_arr,
                )
                affected_routes = [best_move.r_src, best_move.r_dst]
            elif best_move_type == "swap":
                assert isinstance(best_move, MoveSwap)
                apply_swap(
                    curr_sol,
                    route_loads,
                    route_dists,
                    best_move.u,
                    best_move.v,
                    best_move.r_u,
                    best_move.p_u,
                    best_move.r_v,
                    best_move.p_v,
                    self.matrix,
                    self._demands_arr,
                )
                affected_routes = [best_move.r_u, best_move.r_v]
            elif best_move_type == "2opts":
                assert isinstance(best_move, Move2OptStar)
                apply_2opt_star(
                    curr_sol,
                    route_loads,
                    route_dists,
                    best_move.r1,
                    best_move.i,
                    best_move.r2,
                    best_move.j,
                    self.matrix,
                    self._demands_arr,
                )
                affected_routes = [best_move.r1, best_move.r2]

            # Dọn dẹp các tuyến trống chỉ khi có tuyến bị trống thực tế (len <= 2)
            need_clean = False
            for r_idx in affected_routes:
                if len(curr_sol[r_idx]) <= 2:
                    need_clean = True
                    break

            if need_clean:
                clean_empty_routes(curr_sol, route_loads, route_dists)
                pos_map = node_positions(curr_sol)
                suffix_demands = self._build_suffix_demands(curr_sol)
            else:
                self._update_pos_map_for_routes(pos_map, curr_sol, affected_routes)
                for r_idx in affected_routes:
                    suffix_demands[r_idx] = self._compute_route_suffix_demands(curr_sol[r_idx])

            tenure = random.randint(self.tau_min, self.tau_max)
            tabu_dict[best_move_key] = iteration + tenure

            if iteration % 20 == 0:
                tabu_dict = {k: v for k, v in tabu_dict.items() if v >= iteration}

            curr_dist = total_cost_cached(route_dists)
            is_feasible = all(
                route_loads.get(i, 0.0) <= self.capacity
                for i in range(len(curr_sol))
                if len(curr_sol[i]) > 2
            )

            if is_feasible and curr_dist < best_dist:
                best_dist = curr_dist
                best_sol = copy_sol(curr_sol)
                no_improve = 0
                print(
                    f"  [GTS iter {iteration:5d}] ✓ {curr_dist / 100:.2f} km "
                    f"| {len(best_sol)} xe | {best_move_type}"
                )
            else:
                no_improve += 1
                if not is_feasible:
                    infeasible_count += 1

            if iteration % self.pen_h == 0 and iteration > 0:
                ratio = infeasible_count / self.pen_h
                if ratio > 0.5:
                    self.lam *= 1.1
                elif ratio < 0.2:
                    self.lam = max(self.lam * 0.9, 1.0)
                infeasible_count = 0

            if iteration % 200 == 0 and iteration > 0:
                print(
                    f"  [GTS iter {iteration:5d}] best={best_dist / 100:.2f} km "
                    f"| NoImprove={no_improve}/{self.max_no_improve} "
                    f"| λ={self.lam:.1f}"
                )

            iteration += 1

        print(f"\n[GTS] Post-opt Or-opt intra-route...")
        best_sol = self._intra_or_opt(best_sol)
        best_dist = sum(route_dist_raw(r, self.matrix) for r in best_sol if len(r) > 2)
        print(f"[GTS] Hoàn tất: {best_dist / 100:.2f} km | {len(best_sol)} xe")
        return best_sol, best_dist
