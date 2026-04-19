import numpy as np
from typing import Dict, List, Optional, Tuple
import random

Route    = List[int]
Solution = List[Route]


class GranularTabuSearch:

    def __init__(self,
                 distance_matrix: np.ndarray,
                 demands:         Dict[int, float],
                 capacity:        float,
                 max_v:           int,
                 tabu_size:       int   = 20,
                 max_iter:        int   = 3000,
                 max_no_improve:  int   = 400,
                 granular_beta:   float = 1.5,
                 granular_k:      int   = 15,
                 penalty_lambda:  float = None,
                 penalty_h:       int   = 20):

        self.matrix   = distance_matrix
        self.n        = distance_matrix.shape[0]
        self.demands  = demands
        self.capacity = capacity
        self.max_v    = max_v

        self.tabu_size      = tabu_size
        self.max_iter       = max_iter
        self.max_no_improve = max_no_improve
        self.granular_beta  = granular_beta
        self.granular_k     = granular_k
        self.pen_h          = penalty_h

        # [FIX-2] Auto-compute penalty_lambda
        if penalty_lambda is None:
            nonzero  = distance_matrix[distance_matrix > 0]
            avg_edge = float(np.mean(nonzero)) if len(nonzero) > 0 else 100.0
            self.lam = avg_edge / max(capacity, 1.0)
        else:
            nonzero  = distance_matrix[distance_matrix > 0]
            avg_edge = float(np.mean(nonzero)) if len(nonzero) > 0 else 100.0
            auto_lam = avg_edge / max(capacity, 1.0)
            self.lam = max(penalty_lambda, auto_lam * 0.5)

        # [PERF-5] Early-exit threshold: 10% of avg edge cost
        nonzero       = distance_matrix[distance_matrix > 0]
        self._avg_edge = float(np.mean(nonzero)) if len(nonzero) > 0 else 100.0
        self._early_exit_threshold = -self._avg_edge * 0.15

        self.tau_min = max(5,  tabu_size // 2)
        self.tau_max = max(15, tabu_size * 2)

        self._granular_neighbors = self._build_granular_lists()

        # [PERF-1] Demands array for fast vectorized access
        self._demands_arr = np.array(
            [demands.get(i, 0) for i in range(self.n)], dtype=np.float64
        )

    # ── Setup ──────────────────────────────────────────────────────────

    def _build_granular_lists(self) -> Dict[int, List[int]]:
        customers   = list(range(1, self.n))
        depot_dists = self.matrix[0, 1:].astype(float)
        avg_dist    = float(np.mean(depot_dists[depot_dists > 0])) if len(depot_dists) > 0 else 1.0
        threshold   = self.granular_beta * avg_dist * 2

        neighbors: Dict[int, List[int]] = {}
        for i in customers:
            row      = self.matrix[i].astype(float)
            eligible = [j for j in customers if j != i and row[j] <= threshold]
            if len(eligible) < self.granular_k:
                sorted_j = np.argsort(row)
                eligible = [j for j in sorted_j if j != 0 and j != i][:self.granular_k]
            else:
                eligible.sort(key=lambda j: row[j])
                eligible = eligible[:self.granular_k]
            neighbors[i] = eligible

        avg_edges = sum(len(v) for v in neighbors.values()) / max(len(neighbors), 1)
        print(f"[GTS] Granular lists: β={self.granular_beta}, "
              f"avg_nb={avg_edges:.1f}/node, λ={self.lam:.1f}, "
              f"early_exit_thresh={self._early_exit_threshold:.1f}")
        return neighbors

    # ── Cache management ───────────────────────────────────────────────

    def _build_caches(self, sol: Solution) -> Tuple[Dict, Dict]:
        """
        [PERF-1,2,7] Xây dựng route_loads và route_dists cache.
        route_loads[i] = tổng demand của sol[i]
        route_dists[i] = tổng khoảng cách của sol[i]
        """
        route_loads = {}
        route_dists = {}
        for i, r in enumerate(sol):
            route_loads[i] = float(sum(
                self._demands_arr[node] for node in r if node != 0
            ))
            route_dists[i] = self._route_dist_raw(r)
        return route_loads, route_dists

    def _route_dist_raw(self, route: Route) -> float:
        """Tính khoảng cách route từ đầu (dùng khi build cache)."""
        if len(route) <= 2:
            return 0.0
        return float(sum(
            self.matrix[route[i], route[i + 1]]
            for i in range(len(route) - 1)
        ))

    # ── Helpers (dùng cache) ───────────────────────────────────────────

    def _total_cost_cached(self, route_dists: Dict) -> float:
        """[PERF-2] O(num_routes) thay vì O(n)."""
        return sum(route_dists.values())

    def _penalized_cost_cached(self, sol: Solution,
                                route_loads: Dict, route_dists: Dict) -> float:
        dist = penalty = 0.0
        for i, r in enumerate(sol):
            if len(r) <= 2:
                continue
            dist    += route_dists[i]
            penalty += max(0.0, route_loads[i] - self.capacity)
        return dist + self.lam * penalty

    def _copy_sol(self, sol: Solution) -> Solution:
        return [r[:] for r in sol]

    def _node_positions(self, sol: Solution) -> Dict[int, Tuple[int, int]]:
        pos = {}
        for ri, route in enumerate(sol):
            for pi, node in enumerate(route):
                if node != 0:
                    pos[node] = (ri, pi)
        return pos

    def _clean_empty_routes(self, sol: Solution,
                             route_loads: Dict = None,
                             route_dists: Dict = None):
        """
        [FIX-4] Clean routes + đồng bộ cache nếu được truyền vào.
        Rebuild cache keys để liên tục (0,1,2,...).
        """
        keep_indices = [i for i, r in enumerate(sol) if len(r) > 2]
        new_sol = [sol[i] for i in keep_indices]
        sol.clear()
        sol.extend(new_sol)

        if route_loads is not None and route_dists is not None:
            new_loads = {new_i: route_loads[old_i]
                         for new_i, old_i in enumerate(keep_indices)}
            new_dists = {new_i: route_dists[old_i]
                         for new_i, old_i in enumerate(keep_indices)}
            route_loads.clear()
            route_loads.update(new_loads)
            route_dists.clear()
            route_dists.update(new_dists)

    # ── Delta evaluations (dùng cached loads) ──────────────────────────

    def _delta_relocate(self, sol, route_loads, u, r_src, p_u, r_dst, p_ins) -> float:
        route_s = sol[r_src]
        route_d = sol[r_dst]
        d = self.matrix
        prev_u = route_s[p_u - 1]
        next_u = route_s[p_u + 1]
        prev_d = route_d[p_ins - 1]
        next_d = route_d[p_ins]

        remove_gain = d[prev_u, u] + d[u, next_u] - d[prev_u, next_u]
        insert_cost = d[prev_d, u] + d[u, next_d] - d[prev_d, next_d]
        delta       = float(insert_cost - remove_gain)

        u_dem  = self._demands_arr[u]
        load_s = route_loads[r_src]
        load_d = route_loads[r_dst]
        delta += self.lam * (
            max(0.0, load_s - u_dem - self.capacity)
            + max(0.0, load_d + u_dem - self.capacity)
            - max(0.0, load_s - self.capacity)
            - max(0.0, load_d - self.capacity)
        )
        return delta

    def _delta_relocate2(self, sol, route_loads, u, v, r_src, p_u, r_dst, p_ins) -> Optional[float]:
        """[FIX-1] Đúng remove_gain bao gồm d[u,v]."""
        route_s = sol[r_src]
        route_d = sol[r_dst]
        d = self.matrix

        if p_u + 2 >= len(route_s):
            return None
        next_v = route_s[p_u + 2]
        if next_v == 0:
            return None

        prev_u      = route_s[p_u - 1]
        remove_gain = (d[prev_u, u] + d[u, v] + d[v, next_v]
                       - d[prev_u, next_v])
        prev_d      = route_d[p_ins - 1]
        next_d      = route_d[p_ins]
        insert_cost = d[prev_d, u] + d[u, v] + d[v, next_d] - d[prev_d, next_d]
        delta       = float(insert_cost - remove_gain)

        seg_dem = self._demands_arr[u] + self._demands_arr[v]
        load_s  = route_loads[r_src]
        load_d  = route_loads[r_dst]
        delta  += self.lam * (
            max(0.0, load_s - seg_dem - self.capacity)
            + max(0.0, load_d + seg_dem - self.capacity)
            - max(0.0, load_s - self.capacity)
            - max(0.0, load_d - self.capacity)
        )
        return delta

    def _delta_swap(self, sol, route_loads, u, v, r_u, p_u, r_v, p_v) -> Optional[float]:
        route_u = sol[r_u]
        route_v = sol[r_v]
        d       = self.matrix
        pu_prev = route_u[p_u - 1]
        pu_next = route_u[p_u + 1]
        pv_prev = route_v[p_v - 1]
        pv_next = route_v[p_v + 1]

        if r_u == r_v:
            if abs(p_u - p_v) == 1:
                return None
            old = d[pu_prev, u] + d[u, pu_next] + d[pv_prev, v] + d[v, pv_next]
            new = d[pu_prev, v] + d[v, pu_next] + d[pv_prev, u] + d[u, pv_next]
            return float(new - old)
        else:
            old    = d[pu_prev, u] + d[u, pu_next] + d[pv_prev, v] + d[v, pv_next]
            new    = d[pu_prev, v] + d[v, pu_next] + d[pv_prev, u] + d[u, pv_next]
            delta  = float(new - old)
            u_dem  = self._demands_arr[u]
            v_dem  = self._demands_arr[v]
            load_u = route_loads[r_u]
            load_v = route_loads[r_v]
            delta += self.lam * (
                max(0.0, load_u - u_dem + v_dem - self.capacity)
                + max(0.0, load_v - v_dem + u_dem - self.capacity)
                - max(0.0, load_u - self.capacity)
                - max(0.0, load_v - self.capacity)
            )
            return delta

    def _delta_2opt_star(self, sol, route_loads, r1, i, r2, j) -> Optional[float]:
        """[PERF-3] Dùng cached loads thay tính tail từ đầu."""
        route1 = sol[r1]
        route2 = sol[r2]
        d      = self.matrix

        if i == 0 or i >= len(route1) - 1:
            return None
        if j == 0 or j >= len(route2) - 1:
            return None
        # Skip nếu route quá ngắn (< 4 nodes = [0, a, b, 0])
        if len(route1) < 4 or len(route2) < 4:
            return None

        A     = route1[i];  C = route1[i + 1]
        B     = route2[j];  D = route2[j + 1]
        delta = float(d[A, D] + d[B, C] - d[A, C] - d[B, D])

        # [PERF-3] Tính tail_load bằng sum nhanh trên slice (unavoidable)
        # nhưng giới hạn bằng cách chỉ gọi khi delta thô đủ promising
        if delta > self._avg_edge * 2:
            return None  # Không promising, skip luôn

        tail1_load = float(sum(
            self._demands_arr[x] for x in route1[i + 1:] if x != 0
        ))
        tail2_load = float(sum(
            self._demands_arr[x] for x in route2[j + 1:] if x != 0
        ))
        head1_load = route_loads[r1] - tail1_load
        head2_load = route_loads[r2] - tail2_load

        delta += self.lam * (
            max(0.0, head1_load + tail2_load - self.capacity)
            + max(0.0, head2_load + tail1_load - self.capacity)
            - max(0.0, route_loads[r1] - self.capacity)
            - max(0.0, route_loads[r2] - self.capacity)
        )
        return delta

    # ── Apply moves (incremental cache update) ──────────────────────────

    def _apply_relocate(self, sol, route_loads, route_dists, u, r_src, p_u, r_dst, p_ins):
        """[PERF-1,7] Apply + update cache incrementally."""
        route_s = sol[r_src]
        route_d = sol[r_dst]
        d       = self.matrix

        # Tính delta_dist trước khi thay đổi
        prev_u   = route_s[p_u - 1]
        next_u   = route_s[p_u + 1]
        prev_d   = route_d[p_ins - 1]
        next_d   = route_d[p_ins]
        dist_del = (d[prev_u, u] + d[u, next_u]
                    - d[prev_u, next_u])
        dist_ins = (d[prev_d, u] + d[u, next_d]
                    - d[prev_d, next_d])

        # Apply
        route_s.pop(p_u)
        route_d.insert(p_ins, u)

        # Update cache
        u_dem              = self._demands_arr[u]
        route_loads[r_src] -= u_dem
        route_loads[r_dst] += u_dem
        route_dists[r_src] -= dist_del
        route_dists[r_dst] += dist_ins

    def _apply_relocate2(self, sol, route_loads, route_dists, u, v, r_src, p_u, r_dst, p_ins):
        """[FIX-4] + [PERF-1,7] Apply or-opt-2 + update cache."""
        route_s = sol[r_src]
        route_d = sol[r_dst]
        d       = self.matrix

        if p_u + 2 >= len(route_s):
            return
        next_v   = route_s[p_u + 2]
        prev_u   = route_s[p_u - 1]
        prev_d   = route_d[p_ins - 1]
        next_d   = route_d[p_ins]

        dist_del = (d[prev_u, u] + d[u, v] + d[v, next_v]
                    - d[prev_u, next_v])
        dist_ins = (d[prev_d, u] + d[u, v] + d[v, next_d]
                    - d[prev_d, next_d])

        # [FIX-4] Xóa v trước (index cao hơn), rồi u
        route_s.pop(p_u + 1)
        route_s.pop(p_u)
        route_d.insert(p_ins, v)
        route_d.insert(p_ins, u)

        seg_dem            = self._demands_arr[u] + self._demands_arr[v]
        route_loads[r_src] -= seg_dem
        route_loads[r_dst] += seg_dem
        route_dists[r_src] -= dist_del
        route_dists[r_dst] += dist_ins

    def _apply_swap(self, sol, route_loads, route_dists, u, v, r_u, p_u, r_v, p_v):
        """Swap không thay đổi loads nếu cùng route. Cross-route thì update delta."""
        route_u = sol[r_u]
        route_v = sol[r_v]
        d       = self.matrix

        pu_prev = route_u[p_u - 1]; pu_next = route_u[p_u + 1]
        pv_prev = route_v[p_v - 1]; pv_next = route_v[p_v + 1]

        dist_delta = (
            d[pu_prev, v] + d[v, pu_next] + d[pv_prev, u] + d[u, pv_next]
            - d[pu_prev, u] - d[u, pu_next] - d[pv_prev, v] - d[v, pv_next]
        )

        sol[r_u][p_u] = v
        sol[r_v][p_v] = u

        if r_u == r_v:
            route_dists[r_u] += dist_delta
        else:
            # Tính riêng delta cho từng route
            du_delta = (d[pu_prev, v] + d[v, pu_next]
                        - d[pu_prev, u] - d[u, pu_next])
            dv_delta = (d[pv_prev, u] + d[u, pv_next]
                        - d[pv_prev, v] - d[v, pv_next])
            route_dists[r_u] += du_delta
            route_dists[r_v] += dv_delta
            u_dem              = self._demands_arr[u]
            v_dem              = self._demands_arr[v]
            route_loads[r_u]  += v_dem - u_dem
            route_loads[r_v]  += u_dem - v_dem

    def _apply_2opt_star(self, sol, route_loads, route_dists, r1, i, r2, j):
        """[FIX-4] Apply 2opt* + rebuild cache cho 2 routes bị thay đổi."""
        new_r1 = sol[r1][:i + 1] + sol[r2][j + 1:]
        new_r2 = sol[r2][:j + 1] + sol[r1][i + 1:]
        sol[r1] = new_r1
        sol[r2] = new_r2
        # Rebuild cache cho 2 routes này
        route_loads[r1] = float(sum(
            self._demands_arr[node] for node in new_r1 if node != 0
        ))
        route_loads[r2] = float(sum(
            self._demands_arr[node] for node in new_r2 if node != 0
        ))
        route_dists[r1] = self._route_dist_raw(new_r1)
        route_dists[r2] = self._route_dist_raw(new_r2)

    # ── Perturbation ───────────────────────────────────────────────────

    def _double_bridge(self, sol: Solution) -> Solution:
        """[FIX-3] Không recursive; giới hạn k = min(8, len//2)."""
        new_sol = self._copy_sol(sol)
        k       = min(8, len(new_sol) // 2)
        if k < 2:
            return new_sol

        idx_to_shake = random.sample(range(len(new_sol)), k)
        flat = []
        for idx in sorted(idx_to_shake, reverse=True):
            route = new_sol.pop(idx)
            flat.extend([node for node in route if node != 0])

        if len(flat) < 4:
            return self._copy_sol(sol)

        positions = sorted(random.sample(range(1, len(flat)), min(4, len(flat) - 1)))
        while len(positions) < 4:
            positions.append(len(flat))
        a, b, c, _ = positions[:4]

        seg0 = flat[:a]; seg1 = flat[a:b]; seg2 = flat[b:c]; seg3 = flat[c:]
        new_flat = seg0 + seg2 + seg1 + seg3

        shaken = self._rebuild_from_flat(new_flat)
        new_sol.extend(shaken)
        return new_sol

    def _rebuild_from_flat(self, flat: List[int]) -> Solution:
        sol: Solution = []; curr: Route = [0]; load = 0.0
        for node in flat:
            dem = self._demands_arr[node]
            if load + dem > self.capacity:
                curr.append(0); sol.append(curr)
                curr = [0]; load = 0.0
            curr.append(node); load += dem
        curr.append(0); sol.append(curr)
        while len(sol) > self.max_v:
            last = sol.pop()
            sol[-1] = sol[-1][:-1] + last[1:]
        return sol

    # ── Post-optimization ──────────────────────────────────────────────

    def _intra_or_opt(self, sol: Solution) -> Solution:
        """Or-opt nội tuyến làm mịn sau khi tìm best solution."""
        d = self.matrix
        for r in sol:
            if len(r) <= 4:
                continue
            max_passes = 20; pass_count = 0; improved = True
            while improved and pass_count < max_passes:
                improved = False; pass_count += 1; n_r = len(r)
                for i in range(1, n_r - 1):
                    node    = r[i]; prev_i = r[i - 1]; next_i = r[i + 1]
                    gain_rm = d[prev_i, node] + d[node, next_i] - d[prev_i, next_i]
                    best_gain = 1e-6; best_j = -1
                    for j in range(1, n_r - 1):
                        if j == i or j == i - 1:
                            continue
                        prev_j = r[j - 1]; next_j = r[j]
                        gain_ins = d[prev_j, node] + d[node, next_j] - d[prev_j, next_j]
                        gain = gain_rm - gain_ins
                        if gain > best_gain:
                            best_gain = gain; best_j = j
                    if best_j != -1:
                        r.pop(i)
                        ins = best_j if best_j < i else best_j - 1
                        r.insert(ins, node); improved = True
                        n_r = len(r); break
        return sol

    # ── Main loop ──────────────────────────────────────────────────────

    def solve(self, initial_solution: Solution) -> Tuple[Solution, float]:
        curr_sol = self._copy_sol(initial_solution)
        self._clean_empty_routes(curr_sol)

        # [PERF-1,2,7] Khởi tạo cache
        route_loads, route_dists = self._build_caches(curr_sol)

        best_sol  = self._copy_sol(curr_sol)
        best_dist = self._total_cost_cached(route_dists)

        # [PERF-4] Tabu dict thay deque
        tabu_dict: Dict[tuple, int] = {}
        no_improve = 0; iteration = 0; infeasible_count = 0

        print(f"[GTS] Bắt đầu: {len(curr_sol)} xe | "
              f"{best_dist / 100:.2f} km | λ={self.lam:.1f} | "
              f"max_iter={self.max_iter} | max_no_improve={self.max_no_improve}")

        while iteration < self.max_iter:
            if no_improve >= self.max_no_improve:
                print(f"[GTS] Dừng iter {iteration}: "
                      f"{no_improve} vòng không cải thiện")
                break

            # Diversification tại nửa ngưỡng
            if no_improve == self.max_no_improve // 2:
                perturbed = self._double_bridge(self._copy_sol(best_sol))
                self._clean_empty_routes(perturbed)
                p_loads, p_dists = self._build_caches(perturbed)
                p_cost = self._total_cost_cached(p_dists)
                if p_cost < best_dist:
                    best_dist = p_cost
                    best_sol  = self._copy_sol(perturbed)
                    no_improve = 0
                    print(f"  [GTS] Perturbation NEW BEST: {best_dist / 100:.2f} km")
                if p_cost < best_dist * 1.1:
                    curr_sol    = perturbed
                    route_loads = p_loads
                    route_dists = p_dists

            # [FIX-4] + [PERF-8] Rebuild pos_map
            self._clean_empty_routes(curr_sol, route_loads, route_dists)
            pos_map   = self._node_positions(curr_sol)
            base_cost = self._penalized_cost_cached(curr_sol, route_loads, route_dists)

            best_move_delta = float('inf')
            best_move = None; best_move_key = None; best_move_type = None
            found_early_exit = False

            for u in list(pos_map.keys()):
                if found_early_exit:
                    break

                r_u, p_u = pos_map[u]
                route_u  = curr_sol[r_u]

                # ── Or-opt-1 (Relocate single node) ──────────────────
                for v in self._granular_neighbors.get(u, []):
                    if v not in pos_map:
                        continue
                    r_v, p_v = pos_map[v]
                    if r_v == r_u:
                        continue

                    for p_ins in (p_v, p_v + 1):
                        if p_ins < 1 or p_ins >= len(curr_sol[r_v]):
                            continue
                        delta = self._delta_relocate(
                            curr_sol, route_loads, u, r_u, p_u, r_v, p_ins)
                        key   = ('R1', u, r_v)
                        in_tabu = tabu_dict.get(key, -1) >= iteration
                        asp     = (base_cost + delta < best_dist - 1e-6)
                        if (not in_tabu or asp) and delta < best_move_delta:
                            best_move_delta = delta
                            best_move       = (u, r_u, p_u, r_v, p_ins)
                            best_move_key   = key
                            best_move_type  = 'rel1'
                            # [PERF-5] Early exit: cải thiện rõ ràng
                            if delta < self._early_exit_threshold:
                                found_early_exit = True
                                break
                    if found_early_exit:
                        break

                if found_early_exit:
                    break

                # ── Or-opt-2 (Relocate 2 consecutive nodes) ──────────
                if p_u + 1 < len(route_u) - 1:
                    v_next = route_u[p_u + 1]
                    if v_next != 0:
                        for nb in self._granular_neighbors.get(u, [])[:8]:
                            if nb not in pos_map:
                                continue
                            r_nb, p_nb = pos_map[nb]
                            if r_nb == r_u:
                                continue
                            for p_ins in range(1, min(len(curr_sol[r_nb]), 6)):
                                delta2 = self._delta_relocate2(
                                    curr_sol, route_loads,
                                    u, v_next, r_u, p_u, r_nb, p_ins)
                                if delta2 is None:
                                    continue
                                key2    = ('R2', u, r_nb)
                                in_t2   = tabu_dict.get(key2, -1) >= iteration
                                asp2    = (base_cost + delta2 < best_dist - 1e-6)
                                if (not in_t2 or asp2) and delta2 < best_move_delta:
                                    best_move_delta = delta2
                                    best_move       = (u, v_next, r_u, p_u, r_nb, p_ins)
                                    best_move_key   = key2
                                    best_move_type  = 'rel2'
                            break  # Chỉ xét neighbor đầu tiên

                # ── SWAP ─────────────────────────────────────────────
                for v in self._granular_neighbors.get(u, []):
                    if v not in pos_map:
                        continue
                    r_v, p_v = pos_map[v]
                    if r_u == r_v and p_u >= p_v:
                        continue
                    ds = self._delta_swap(
                        curr_sol, route_loads, u, v, r_u, p_u, r_v, p_v)
                    if ds is None:
                        continue
                    keys    = ('SW', min(u, v), max(u, v))
                    in_t_s  = tabu_dict.get(keys, -1) >= iteration
                    asp_s   = (base_cost + ds < best_dist - 1e-6)
                    if (not in_t_s or asp_s) and ds < best_move_delta:
                        best_move_delta = ds
                        best_move       = (u, v, r_u, p_u, r_v, p_v)
                        best_move_key   = keys
                        best_move_type  = 'swap'
                        if ds < self._early_exit_threshold:
                            found_early_exit = True
                            break

                if found_early_exit:
                    break

                # ── 2-opt* ───────────────────────────────────────────
                for v in self._granular_neighbors.get(u, [])[:6]:
                    if v not in pos_map:
                        continue
                    r_v, p_v = pos_map[v]
                    if r_v == r_u:
                        continue
                    d2s = self._delta_2opt_star(
                        curr_sol, route_loads, r_u, p_u, r_v, p_v)
                    if d2s is None:
                        continue
                    key2s   = ('2S', r_u, p_u, r_v, p_v)
                    in_t_2s = tabu_dict.get(key2s, -1) >= iteration
                    asp_2s  = (base_cost + d2s < best_dist - 1e-6)
                    if (not in_t_2s or asp_2s) and d2s < best_move_delta:
                        best_move_delta = d2s
                        best_move       = (r_u, p_u, r_v, p_v)
                        best_move_key   = key2s
                        best_move_type  = '2opts'

            # Apply best move
            if best_move is None:
                no_improve += 1; iteration += 1
                continue

            if best_move_type == 'rel1':
                u, r_src, p_u, r_dst, p_ins = best_move
                self._apply_relocate(
                    curr_sol, route_loads, route_dists,
                    u, r_src, p_u, r_dst, p_ins)
            elif best_move_type == 'rel2':
                u, v_nxt, r_src, p_u, r_dst, p_ins = best_move
                self._apply_relocate2(
                    curr_sol, route_loads, route_dists,
                    u, v_nxt, r_src, p_u, r_dst, p_ins)
            elif best_move_type == 'swap':
                u, v, r_u, p_u, r_v, p_v = best_move
                self._apply_swap(
                    curr_sol, route_loads, route_dists,
                    u, v, r_u, p_u, r_v, p_v)
            elif best_move_type == '2opts':
                r1, i, r2, j = best_move
                self._apply_2opt_star(
                    curr_sol, route_loads, route_dists, r1, i, r2, j)

            # [FIX-4] Clean sau apply
            self._clean_empty_routes(curr_sol, route_loads, route_dists)

            # [PERF-4] Tabu update với random tenure
            tenure = random.randint(self.tau_min, self.tau_max)
            tabu_dict[best_move_key] = iteration + tenure

            # [PERF-4] Clean aggressive hơn: mỗi 20 iter
            if iteration % 20 == 0:
                tabu_dict = {k: v for k, v in tabu_dict.items()
                             if v >= iteration}

            # Check improvement
            curr_dist   = self._total_cost_cached(route_dists)
            is_feasible = all(
                route_loads.get(i, 0) <= self.capacity
                for i in range(len(curr_sol))
                if len(curr_sol[i]) > 2
            )

            if is_feasible and curr_dist < best_dist:
                best_dist  = curr_dist
                best_sol   = self._copy_sol(curr_sol)
                no_improve = 0
                print(f"  [GTS iter {iteration:5d}] ✓ {curr_dist / 100:.2f} km "
                      f"| {len(best_sol)} xe | {best_move_type}")
            else:
                no_improve += 1
                if not is_feasible:
                    infeasible_count += 1

            # Adaptive penalty
            if iteration % self.pen_h == 0 and iteration > 0:
                ratio = infeasible_count / self.pen_h
                if ratio > 0.5:
                    self.lam *= 1.1
                elif ratio < 0.2:
                    self.lam = max(self.lam * 0.9, 1.0)
                infeasible_count = 0

            if iteration % 200 == 0 and iteration > 0:
                print(f"  [GTS iter {iteration:5d}] best={best_dist / 100:.2f} km "
                      f"| NoImprove={no_improve}/{self.max_no_improve} "
                      f"| λ={self.lam:.1f}")

            iteration += 1

        print(f"\n[GTS] Post-opt Or-opt intra-route...")
        best_sol  = self._intra_or_opt(best_sol)
        best_dist = self._route_dist_raw.__func__(  # recalc after 2opt
            self,
            [node for r in best_sol for node in r]  # dummy, use loop below
        ) if False else sum(self._route_dist_raw(r) for r in best_sol if len(r) > 2)
        print(f"[GTS] Hoàn tất: {best_dist / 100:.2f} km | {len(best_sol)} xe")
        return best_sol, best_dist