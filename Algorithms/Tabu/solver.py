# Lớp điều phối chính thực thi giải thuật tối ưu hóa Granular Tabu Search (GTS).
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from Utils.Operators.local_search import merge_excess_routes_safe, or_opt_intra, build_granular_lists
from Algorithms.Tabu.structures import (
    AnyMove, Move2OptStar, MoveRel1, MoveRel2, MoveSwap, Route, Solution,
)
from Algorithms.Tabu.utils import (
    build_caches, clean_empty_routes,
    copy_sol, node_positions, penalized_cost_cached,
    route_dist_raw, total_cost_cached,
)
from Algorithms.Tabu.move_evaluator import (
    eval_2opt_star, eval_relocate, eval_relocate2, eval_swap,
)
from Algorithms.Tabu.move_applier import (
    apply_2opt_star, apply_relocate, apply_relocate2, apply_swap,
)


@dataclass
class SearchContext:
    """Gom nhóm trạng thái tìm kiếm vòng lặp GTS tránh truyền từng tham số rời rạc."""
    pos_map:        dict
    curr_sol:       list
    route_loads:    dict
    route_dists:    dict
    suffix_demands: dict
    base_cost:      float
    best_dist:      float
    iteration:      int
    tabu_dict:      dict = field(default_factory=dict)
    best_move_delta: float = float("inf")


# Dispatch dict thay thế if/elif chain để áp dụng move theo loại.
MOVE_APPLIERS = {
    MoveRel1:     (apply_relocate,
                   lambda m: (m.u, m.r_src, m.p_u, m.r_dst, m.p_ins),
                   lambda m: [m.r_src, m.r_dst]),
    MoveRel2:     (apply_relocate2,
                   lambda m: (m.u, m.v_nxt, m.r_src, m.p_u, m.r_dst, m.p_ins),
                   lambda m: [m.r_src, m.r_dst]),
    MoveSwap:     (apply_swap,
                   lambda m: (m.u, m.v, m.r_u, m.p_u, m.r_v, m.p_v),
                   lambda m: [m.r_u, m.r_v]),
    Move2OptStar: (apply_2opt_star,
                   lambda m: (m.r1, m.i, m.r2, m.j),
                   lambda m: [m.r1, m.r2]),
}


class PenaltyController:
    """Điều phối hệ số phạt vi phạm ràng buộc tải trọng thích nghi (Adaptive Penalty)."""

    def __init__(self, init_lambda: float, penalty_h: int = 20) -> None:
        # Khởi tạo hệ số phạt lam ban đầu và chu kỳ thích nghi.
        self.lam = init_lambda
        self.pen_h = penalty_h
        self.infeasible_count = 0

    def register_state(self, is_feasible: bool) -> None:
        # Đăng ký tính khả thi tải trọng của nghiệm tại vòng lặp hiện tại.
        if not is_feasible:
            self.infeasible_count += 1

    def update_penalty(self, iteration: int) -> float:
        # Cập nhật và điều chỉnh hệ số phạt lam thích nghi dựa trên tỷ lệ vi phạm trong chu kỳ.
        if iteration % self.pen_h == 0 and iteration > 0:
            ratio = self.infeasible_count / self.pen_h
            if ratio > 0.5:
                self.lam *= 1.1
            elif ratio < 0.2:
                self.lam = max(self.lam * 0.9, 1.0)
            self.infeasible_count = 0
        return self.lam


class TabuSearchLogger:
    """Ghi log tiến trình thực thi của giải thuật Granular Tabu Search ra console."""

    def __init__(self, verbose: bool = True) -> None:
        # Thiết lập cờ verbose để bật/tắt ghi log ra màn hình console.
        self.verbose = verbose

    def log_start(self, num_vehicles: int, init_dist: float, lam: float, max_iter: int, max_no_improve: int) -> None:
        # Ghi nhận log khởi hành thuật toán tối ưu hóa GTS.
        if self.verbose:
            print(f"[GTS] Bắt đầu: {num_vehicles} xe | {init_dist / 100:.2f} km | λ={lam:.1f} | "
                  f"max_iter={max_iter} | max_no_improve={max_no_improve}")

    def log_new_best(self, iteration: int, curr_dist: float, num_vehicles: int, move_type: str) -> None:
        # Ghi nhận log khi tìm thấy lời giải tốt nhất khả thi mới.
        if self.verbose:
            print(f"  [GTS iter {iteration:5d}] ✓ {curr_dist / 100:.2f} km | {num_vehicles} xe | {move_type}")

    def log_perturbation_best(self, best_dist: float) -> None:
        # Ghi nhận log khi bước perturbation tìm được nghiệm tốt hơn mốc cũ.
        if self.verbose:
            print(f"  [GTS] Perturbation NEW BEST: {best_dist / 100:.2f} km")

    def log_period(self, iteration: int, best_dist: float, no_improve: int, max_no_improve: int, lam: float) -> None:
        # Ghi nhận log định kỳ thông số tối ưu hiện tại.
        if self.verbose:
            print(f"  [GTS iter {iteration:5d}] best={best_dist / 100:.2f} km | "
                  f"NoImprove={no_improve}/{max_no_improve} | λ={lam:.1f}")

    def log_stop(self, iteration: int, no_improve: int) -> None:
        # Ghi nhận log dừng tối ưu sớm khi không cải thiện.
        if self.verbose:
            print(f"[GTS] Dừng iter {iteration}: {no_improve} vòng không cải thiện")

    def log_finalize(self, best_dist: float, num_vehicles: int) -> None:
        # Ghi nhận log hoàn tất thuật toán và tổng kết kết quả.
        if self.verbose:
            print(f"[GTS] Hoàn tất: {best_dist / 100:.2f} km | {num_vehicles} xe")


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
        logger:          Optional[TabuSearchLogger] = None,
        penalty_controller: Optional[PenaltyController] = None,
    ):
        # Khởi tạo các tham số GTS, ma trận khoảng cách, các ràng buộc và candidate list.
        self.matrix   = distance_matrix
        self.n        = distance_matrix.shape[0]
        self.demands  = demands
        self.capacity = capacity
        self.max_v    = max_v

        self.tabu_size      = tabu_size
        self.max_iter       = max_iter
        self.max_no_improve = max_no_improve
        self.pen_h          = penalty_h

        self.logger = logger if logger is not None else TabuSearchLogger(verbose=True)

        nonzero   = distance_matrix[distance_matrix > 0]
        avg_edge  = float(np.mean(nonzero)) if len(nonzero) > 0 else 100.0
        auto_lam  = avg_edge / max(capacity, 1.0)
        init_lam  = max(penalty_lambda, auto_lam * 0.5) if penalty_lambda else auto_lam

        self.penalty_controller = penalty_controller if penalty_controller is not None else PenaltyController(init_lam, penalty_h)
        self.lam  = self.penalty_controller.lam

        self._avg_edge              = avg_edge
        self._early_exit_threshold  = -avg_edge * 0.15
        self.tau_min                = max(5, tabu_size // 2)
        self.tau_max                = max(15, tabu_size * 2)

        self._granular_neighbors = build_granular_lists(
            self.matrix, self.n, granular_beta, granular_k
        )
        self._demands_arr = np.array(
            [demands.get(i, 0.0) for i in range(self.n)], dtype=np.float64
        )

        avg_nb = sum(len(v) for v in self._granular_neighbors.values()) / max(len(self._granular_neighbors), 1)
        self.logger.log_start(max_v, 0.0, self.lam, max_iter, max_no_improve)
        print(f"[GTS-Config] Khởi tạo GTS: β={granular_beta}, avg_nb={avg_nb:.1f}/node, early_exit_thresh={self._early_exit_threshold:.1f}")

    # ──────────────────── Tiện ích nội bộ ────────────────────

    def _double_bridge(self, sol: Solution) -> Solution:
        # Thực hiện đột phá nghiệm (shake) bằng phép biến đổi Double-Bridge.
        new_sol = copy_sol(sol)
        k = min(8, len(new_sol) // 2)
        if k < 2:
            return new_sol

        idx_to_shake = random.sample(range(len(new_sol)), k)
        flat = []
        for idx in sorted(idx_to_shake, reverse=True):
            flat.extend(n for n in new_sol.pop(idx) if n != 0)

        if len(flat) < 4:
            return copy_sol(sol)

        positions = sorted(random.sample(range(1, len(flat)), min(4, len(flat) - 1)))
        while len(positions) < 4:
            positions.append(len(flat))
        a, b, c, _ = positions[:4]
        new_flat = flat[:a] + flat[b:c] + flat[a:b] + flat[c:]

        new_sol.extend(self._rebuild_from_flat(new_flat))
        return new_sol

    def _rebuild_from_flat(self, flat: List[int]) -> Solution:
        # Dựng lại phương án hợp lệ từ danh sách phẳng khách hàng.
        sol, curr, load = [], [0], 0.0
        for node in flat:
            dem = self._demands_arr[node]
            if load + dem > self.capacity:
                curr.append(0); sol.append(curr); curr = [0]; load = 0.0
            curr.append(node); load += dem
        curr.append(0); sol.append(curr)
        return merge_excess_routes_safe(sol, self.max_v, self.demands, self.capacity)

    def _intra_or_opt(self, sol: Solution) -> Solution:
        # Tối ưu hóa cục bộ nội tuyến Or-opt-1 cho toàn bộ tuyến đường sau tìm kiếm.
        for idx, r in enumerate(sol):
            sol[idx] = or_opt_intra(self.matrix, r)
        return sol

    def _compute_route_suffix_demands(self, route: Route) -> List[float]:
        # Tính mảng suffix demands tích lũy ngược cho một tuyến đường.
        n_r, suffixes, val = len(route), [0.0] * len(route), 0.0
        for idx in range(n_r - 1, -1, -1):
            if route[idx] != 0:
                val += self._demands_arr[route[idx]]
            suffixes[idx] = val
        return suffixes

    def _build_suffix_demands(self, sol: Solution) -> Dict[int, List[float]]:
        # Xây dựng mảng suffix demands cho toàn bộ các tuyến đường trong nghiệm.
        return {r_idx: self._compute_route_suffix_demands(r) for r_idx, r in enumerate(sol)}

    def _update_pos_map_for_routes(self, pos_map: dict, curr_sol: list, route_indices: list):
        # Cập nhật gia tăng pos_map chỉ cho các tuyến bị ảnh hưởng bởi nước đi vừa thực hiện.
        for r_idx in route_indices:
            for p_idx, node in enumerate(curr_sol[r_idx]):
                if node != 0:
                    pos_map[node] = (r_idx, p_idx)

    # ──────────────────── Tìm kiếm lân cận ────────────────────

    def _explore_relocate1(self, u, r_u, p_u, remove_gain, ctx: SearchContext) -> tuple:
        # Tìm nước đi Relocate 1 node tốt nhất trong danh sách lân cận granular.
        local_delta, local_move, local_key, early_exit = ctx.best_move_delta, None, None, False
        for v in self._granular_neighbors.get(u, []):
            if v not in ctx.pos_map: continue
            r_v, p_v = ctx.pos_map[v]
            if r_v == r_u: continue
            for p_ins in (p_v, p_v + 1):
                if p_ins < 1 or p_ins >= len(ctx.curr_sol[r_v]): continue
                delta = eval_relocate(ctx.curr_sol, ctx.route_loads, u, r_u, p_u, r_v, p_ins,
                                      self.matrix, self._demands_arr, self.capacity, self.lam, remove_gain)
                key   = ("R1", u, r_v)
                in_tabu = ctx.tabu_dict.get(key, -1) >= ctx.iteration
                asp     = ctx.base_cost + delta < ctx.best_dist - 1e-6
                if (not in_tabu or asp) and delta < local_delta:
                    local_delta, local_move, local_key = delta, MoveRel1(u, r_u, p_u, r_v, p_ins), key
                    if delta < self._early_exit_threshold: early_exit = True; break
            if early_exit: break
        return local_delta, local_move, local_key, early_exit

    def _explore_relocate2(self, u, v_next, r_u, p_u, remove_gain, ctx: SearchContext) -> tuple:
        # Tìm nước đi Relocate 2 nodes tốt nhất trong danh sách lân cận granular.
        local_delta, local_move, local_key = ctx.best_move_delta, None, None
        for nb in self._granular_neighbors.get(u, [])[:8]:
            if nb not in ctx.pos_map: continue
            r_nb, _ = ctx.pos_map[nb]
            if r_nb == r_u: continue
            for p_ins in range(1, min(len(ctx.curr_sol[r_nb]), 6)):
                delta2 = eval_relocate2(ctx.curr_sol, ctx.route_loads, u, v_next, r_u, p_u, r_nb, p_ins,
                                        self.matrix, self._demands_arr, self.capacity, self.lam, remove_gain)
                if delta2 is None: continue
                key2    = ("R2", u, r_nb)
                in_tabu = ctx.tabu_dict.get(key2, -1) >= ctx.iteration
                asp2    = ctx.base_cost + delta2 < ctx.best_dist - 1e-6
                if (not in_tabu or asp2) and delta2 < local_delta:
                    local_delta, local_move, local_key = delta2, MoveRel2(u, v_next, r_u, p_u, r_nb, p_ins), key2
            break
        return local_delta, local_move, local_key

    def _explore_swap(self, u, r_u, p_u, ctx: SearchContext) -> tuple:
        # Tìm nước đi Swap tốt nhất giữa hai khách hàng trong danh sách lân cận.
        local_delta, local_move, local_key, early_exit = ctx.best_move_delta, None, None, False
        for v in self._granular_neighbors.get(u, []):
            if v not in ctx.pos_map: continue
            r_v, p_v = ctx.pos_map[v]
            if r_u == r_v and p_u >= p_v: continue
            ds = eval_swap(ctx.curr_sol, ctx.route_loads, u, v, r_u, p_u, r_v, p_v,
                           self.matrix, self._demands_arr, self.capacity, self.lam)
            if ds is None: continue
            key     = ("SW", min(u, v), max(u, v))
            in_tabu = ctx.tabu_dict.get(key, -1) >= ctx.iteration
            asp     = ctx.base_cost + ds < ctx.best_dist - 1e-6
            if (not in_tabu or asp) and ds < local_delta:
                local_delta, local_move, local_key = ds, MoveSwap(u, v, r_u, p_u, r_v, p_v), key
                if ds < self._early_exit_threshold: early_exit = True; break
        return local_delta, local_move, local_key, early_exit

    def _explore_2optstar(self, u, r_u, p_u, ctx: SearchContext) -> tuple:
        # Tìm nước đi 2-opt* tốt nhất sử dụng suffix_demands O(1) để kiểm tra tải trọng.
        local_delta, local_move, local_key = ctx.best_move_delta, None, None
        for v in self._granular_neighbors.get(u, [])[:6]:
            if v not in ctx.pos_map: continue
            r_v, p_v = ctx.pos_map[v]
            if r_v == r_u: continue
            d2s = eval_2opt_star(ctx.curr_sol, ctx.route_loads, r_u, p_u, r_v, p_v,
                                 self.matrix, ctx.suffix_demands, self.capacity, self.lam, self._avg_edge)
            if d2s is None: continue
            key     = ("2S", r_u, p_u, r_v, p_v)
            in_tabu = ctx.tabu_dict.get(key, -1) >= ctx.iteration
            asp     = ctx.base_cost + d2s < ctx.best_dist - 1e-6
            if (not in_tabu or asp) and d2s < local_delta:
                local_delta, local_move, local_key = d2s, Move2OptStar(r_u, p_u, r_v, p_v), key
        return local_delta, local_move, local_key

    # ──────────────────── Vòng lặp chính ────────────────────

    def _init_search_state(self, initial_solution: Solution) -> tuple:
        # Khởi tạo toàn bộ cache, pos_map và suffix_demands trước khi vào vòng lặp tìm kiếm.
        curr_sol = copy_sol(initial_solution)
        clean_empty_routes(curr_sol)
        route_loads, route_dists = build_caches(curr_sol, self._demands_arr, self.matrix)
        best_sol  = copy_sol(curr_sol)
        best_dist = total_cost_cached(route_dists)
        pos_map   = node_positions(curr_sol)
        suffix_demands = self._build_suffix_demands(curr_sol)
        return curr_sol, route_loads, route_dists, best_sol, best_dist, pos_map, suffix_demands

    def _should_stop(self, no_improve: int) -> bool:
        # Kiểm tra điều kiện dừng sớm khi đạt giới hạn vòng không cải thiện.
        return no_improve >= self.max_no_improve

    def _maybe_perturb(self, ctx: SearchContext, best_sol, best_dist, no_improve, route_dists):
        # Thực hiện đột phá nghiệm (shake) bằng Double-Bridge khi tối ưu hóa bị bế tắc.
        if no_improve != self.max_no_improve // 2:
            return ctx, best_sol, best_dist, no_improve, route_dists

        perturbed = self._double_bridge(copy_sol(best_sol))
        clean_empty_routes(perturbed)
        p_loads, p_dists = build_caches(perturbed, self._demands_arr, self.matrix)
        p_cost = total_cost_cached(p_dists)

        if p_cost < best_dist:
            best_dist = p_cost
            best_sol  = copy_sol(perturbed)
            no_improve = 0
            self.logger.log_perturbation_best(best_dist)

        if p_cost < best_dist * 1.1:
            ctx.curr_sol       = perturbed
            ctx.route_loads    = p_loads
            ctx.route_dists    = p_dists
            ctx.pos_map        = node_positions(perturbed)
            ctx.suffix_demands = self._build_suffix_demands(perturbed)
            route_dists = p_dists

        return ctx, best_sol, best_dist, no_improve, route_dists

    def _find_best_move(self, ctx: SearchContext) -> tuple:
        # Duyệt qua danh sách lân cận granular tìm nước đi tốt nhất trong vòng lặp hiện tại.
        best_move, best_key = None, None
        ctx.best_move_delta = float("inf")
        found_early_exit = False

        for u in list(ctx.pos_map.keys()):
            if found_early_exit: break
            r_u, p_u  = ctx.pos_map[u]
            route_u   = ctx.curr_sol[r_u]
            prev_u, next_u = route_u[p_u - 1], route_u[p_u + 1]
            rg1 = self.matrix[prev_u, u] + self.matrix[u, next_u] - self.matrix[prev_u, next_u]

            d, m, k, early = self._explore_relocate1(u, r_u, p_u, rg1, ctx)
            if m: ctx.best_move_delta, best_move, best_key = d, m, k
            found_early_exit = early
            if found_early_exit: break

            if p_u + 1 < len(route_u) - 1:
                v_next = route_u[p_u + 1]
                if v_next != 0:
                    next_v = route_u[p_u + 2]
                    rg2 = (self.matrix[prev_u, u] + self.matrix[u, v_next]
                           + self.matrix[v_next, next_v] - self.matrix[prev_u, next_v])
                    d, m, k = self._explore_relocate2(u, v_next, r_u, p_u, rg2, ctx)
                    if m: ctx.best_move_delta, best_move, best_key = d, m, k

            d, m, k, early = self._explore_swap(u, r_u, p_u, ctx)
            if m: ctx.best_move_delta, best_move, best_key = d, m, k
            found_early_exit = early
            if found_early_exit: break

            d, m, k = self._explore_2optstar(u, r_u, p_u, ctx)
            if m: ctx.best_move_delta, best_move, best_key = d, m, k

        return best_move, best_key

    def _apply_move(self, ctx: SearchContext, best_move: AnyMove):
        # Áp dụng nước đi tốt nhất lên nghiệm hiện tại và cập nhật gia tăng các cache.
        applier_fn, args_fn, routes_fn = MOVE_APPLIERS[type(best_move)]
        applier_fn(ctx.curr_sol, ctx.route_loads, ctx.route_dists,
                   *args_fn(best_move), self.matrix, self._demands_arr)
        affected = routes_fn(best_move)

        need_clean = any(len(ctx.curr_sol[r]) <= 2 for r in affected)
        if need_clean:
            clean_empty_routes(ctx.curr_sol, ctx.route_loads, ctx.route_dists)
            ctx.pos_map        = node_positions(ctx.curr_sol)
            ctx.suffix_demands = self._build_suffix_demands(ctx.curr_sol)
        else:
            self._update_pos_map_for_routes(ctx.pos_map, ctx.curr_sol, affected)
            for r_idx in affected:
                ctx.suffix_demands[r_idx] = self._compute_route_suffix_demands(ctx.curr_sol[r_idx])

        return affected

    def _update_tabu_and_penalty(self, ctx: SearchContext, best_key) -> bool:
        # Cập nhật danh sách tabu và điều chỉnh hệ số phạt lam thích nghi.
        tenure = random.randint(self.tau_min, self.tau_max)
        ctx.tabu_dict[best_key] = ctx.iteration + tenure

        if ctx.iteration % 20 == 0:
            ctx.tabu_dict = {k: v for k, v in ctx.tabu_dict.items() if v >= ctx.iteration}

        is_feasible = all(ctx.route_loads.get(i, 0.0) <= self.capacity
                          for i in range(len(ctx.curr_sol)) if len(ctx.curr_sol[i]) > 2)
        
        self.penalty_controller.register_state(is_feasible)
        self.lam = self.penalty_controller.update_penalty(ctx.iteration)

        return is_feasible

    def _finalize(self, best_sol: Solution, best_dist: float) -> Tuple[Solution, float]:
        # Thực hiện hậu tối ưu Or-opt nội tuyến và trả về nghiệm tốt nhất cuối cùng.
        best_sol  = self._intra_or_opt(best_sol)
        best_dist = sum(route_dist_raw(r, self.matrix) for r in best_sol if len(r) > 2)
        self.logger.log_finalize(best_dist, len(best_sol))
        return best_sol, best_dist

    def solve(self, initial_solution: Solution) -> Tuple[Solution, float]:
        # Thực hiện vòng lặp GTS tìm kiếm lời giải tối ưu theo Template Method đã phân tách.
        curr_sol, route_loads, route_dists, best_sol, best_dist, pos_map, suffix_demands = \
            self._init_search_state(initial_solution)

        self.logger.log_start(len(curr_sol), best_dist, self.lam, self.max_iter, self.max_no_improve)

        tabu_dict:       Dict[tuple, int] = {}
        no_improve       = 0
        iteration        = 0

        while iteration < self.max_iter:
            if self._should_stop(no_improve):
                self.logger.log_stop(iteration, no_improve)
                break

            ctx = SearchContext(
                pos_map=pos_map, curr_sol=curr_sol, route_loads=route_loads,
                route_dists=route_dists, suffix_demands=suffix_demands,
                base_cost=penalized_cost_cached(curr_sol, route_loads, route_dists, self.capacity, self.lam),
                best_dist=best_dist, iteration=iteration, tabu_dict=tabu_dict,
            )

            ctx, best_sol, best_dist, no_improve, route_dists = \
                self._maybe_perturb(ctx, best_sol, best_dist, no_improve, route_dists)

            best_move, best_key = self._find_best_move(ctx)

            if best_move is None:
                no_improve += 1; iteration += 1; continue

            self._apply_move(ctx, best_move)

            is_feasible = self._update_tabu_and_penalty(ctx, best_key)

            curr_dist = total_cost_cached(ctx.route_dists)
            if is_feasible and curr_dist < best_dist:
                best_dist  = curr_dist
                best_sol   = copy_sol(ctx.curr_sol)
                no_improve = 0
                self.logger.log_new_best(iteration, curr_dist, len(best_sol), type(best_move).__name__)
            else:
                no_improve += 1

            if iteration % 200 == 0 and iteration > 0:
                self.logger.log_period(iteration, best_dist, no_improve, self.max_no_improve, self.lam)

            # Đồng bộ lại các biến local từ context sau mỗi bước
            curr_sol, route_loads, pos_map, suffix_demands = \
                ctx.curr_sol, ctx.route_loads, ctx.pos_map, ctx.suffix_demands
            iteration += 1

        return self._finalize(best_sol, best_dist)
