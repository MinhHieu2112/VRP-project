"""
Algorithms/Tabu/tabu_solver.py — FIXED
=======================================
Granular Tabu Search cho ACVRP.
Tham chiếu: Toth & Vigo (2003), Gendreau et al. (1994).

BUGS ĐÃ FIX:
[FIX-1] Or-opt-2 remove_gain THIẾU d[u,v]:
        Cũ: remove_gain = d[prev_u,u] + d[v,next_v] - d[prev_u,next_v]
        Mới: remove_gain = d[prev_u,u] + d[u,v] + d[v,next_v] - d[prev_u,next_v]
        Thiếu d[u,v] → delta bị overestimate → moves cải thiện bị từ chối sai.

[FIX-2] penalty_lambda quá nhỏ so với đơn vị matrix:
        Cũ: default lam=1.0 (= 10m trong hệ đơn vị)
        Mới: lam được init = avg_edge_cost / capacity (≈ 100-200 units)
        Penalty 1.0 đối với matrix đơn vị 10m là không có tác dụng —
        bất kỳ move nào cũng tệ hơn penalty → infeasible search vô nghĩa.

[FIX-3] _double_bridge infinite recursion:
        Cũ: nếu len(sol) < num_routes_to_shake → gọi lại _double_bridge(sol)
            với cùng default=15 → infinite recursion khi ít xe.
        Mới: num_routes_to_shake = min(k, len(sol)//2, 8).

[FIX-4] sol[:] = filter trong _apply methods → stale route indices:
        Cũ: _apply_relocate/relocate2/2opt_star đều gọi
            sol[:] = [r for r in sol if len(r) > 2] → xóa routes rỗng,
            nhưng các indices r_src/r_dst/r1/r2 trong best_move đã tính
            trên new_sol = _copy_sol(curr_sol) → chưa bị filter → ổn.
            Tuy nhiên _apply_2opt_star tạo 2 route mới rồi filter → 
            route indices r1/r2 có thể trỏ sai sau filter nếu route trước
            r1/r2 bị xóa.
        Mới: Không filter trong _apply, luôn rebuild pos_map từ đầu iter.
             Chỉ filter khi rebuild sol ở đầu iteration tiếp theo.

[FIX-5] config_tabu.json: penalty_lambda tương thích với đơn vị matrix.
        Thêm gợi ý trong __init__ docstring.
"""

import numpy as np
from collections import deque
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
                 max_iter:        int   = 10_000,
                 max_no_improve:  int   = 500,
                 granular_beta:   float = 1.5,
                 granular_k:      int   = 20,
                 penalty_lambda:  float = None,   # [FIX-2] None → auto-compute
                 penalty_h:       int   = 10):

        self.matrix   = distance_matrix
        self.n        = distance_matrix.shape[0]
        self.demands  = demands
        self.capacity = capacity
        self.max_v    = max_v

        self.tabu_size      = tabu_size
        self.max_iter       = max_iter
        self.max_no_improve = max_no_improve

        self.granular_beta = granular_beta
        self.granular_k    = granular_k
        self.pen_h         = penalty_h

        # [FIX-2] Auto-compute penalty_lambda nếu không được truyền vào.
        # Công thức: avg_nonzero_edge / capacity.
        # Với matrix đơn vị 10m, avg_edge ~100 units, capacity=10
        # → lam_auto ≈ 10 đủ để penalty 1 đơn vị vi phạm > avg savings.
        if penalty_lambda is None:
            nonzero = distance_matrix[distance_matrix > 0]
            avg_edge = float(np.mean(nonzero)) if len(nonzero) > 0 else 100.0
            self.lam = avg_edge / max(capacity, 1.0)
        else:
            # Nếu truyền vào từ config (thường =1.0), scale lên
            # vì config được viết khi chưa biết đơn vị matrix
            nonzero = distance_matrix[distance_matrix > 0]
            avg_edge = float(np.mean(nonzero)) if len(nonzero) > 0 else 100.0
            auto_lam = avg_edge / max(capacity, 1.0)
            # Nếu penalty_lambda quá nhỏ so với auto, dùng auto
            self.lam = max(penalty_lambda, auto_lam * 0.5)

        self.tau_min = max(5,  tabu_size // 2)
        self.tau_max = max(15, tabu_size * 2)

        self._granular_neighbors = self._build_granular_lists()

    # ── Setup ──────────────────────────────────────────────────────────

    def _build_granular_lists(self) -> Dict[int, List[int]]:
        customers = list(range(1, self.n))
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
        print(f"[GTS] Granular lists: β={self.granular_beta}, avg_nb={avg_edges:.1f}/node, λ={self.lam:.1f}")
        return neighbors

    # ── Helpers ────────────────────────────────────────────────────────

    def _route_dist(self, route: Route) -> float:
        if len(route) <= 2: return 0.0
        return float(sum(self.matrix[route[i], route[i+1]] for i in range(len(route)-1)))

    def _route_load(self, route: Route) -> float:
        return float(sum(self.demands.get(n, 0) for n in route if n != 0))

    def _total_cost(self, sol: Solution) -> float:
        return float(sum(self._route_dist(r) for r in sol if len(r) > 2))

    def _penalized_cost(self, sol: Solution) -> float:
        dist = penalty = 0.0
        for r in sol:
            if len(r) <= 2: continue
            dist    += self._route_dist(r)
            penalty += max(0.0, self._route_load(r) - self.capacity)
        return dist + self.lam * penalty

    def _copy_sol(self, sol: Solution) -> Solution:
        return [r[:] for r in sol]

    def _node_positions(self, sol: Solution) -> Dict[int, Tuple[int,int]]:
        pos = {}
        for ri, route in enumerate(sol):
            for pi, node in enumerate(route):
                if node != 0:
                    pos[node] = (ri, pi)
        return pos

    def _clean_empty_routes(self, sol: Solution):
        """Xóa routes rỗng in-place."""
        sol[:] = [r for r in sol if len(r) > 2]

    # ── Delta evaluations ──────────────────────────────────────────────

    def _delta_relocate(self, sol, u, r_src, p_u, r_dst, p_ins) -> Optional[float]:
        route_s = sol[r_src]; route_d = sol[r_dst]
        d = self.matrix
        prev_u = route_s[p_u-1]; next_u = route_s[p_u+1]
        prev_d = route_d[p_ins-1]; next_d = route_d[p_ins]

        remove_gain = d[prev_u,u] + d[u,next_u] - d[prev_u,next_u]
        insert_cost = d[prev_d,u] + d[u,next_d] - d[prev_d,next_d]
        delta = insert_cost - remove_gain

        u_dem = self.demands.get(u, 0)
        load_s = self._route_load(route_s)
        load_d = self._route_load(route_d)
        delta += self.lam * (
            max(0.0, load_s - u_dem - self.capacity) +
            max(0.0, load_d + u_dem - self.capacity) -
            max(0.0, load_s - self.capacity) -
            max(0.0, load_d - self.capacity)
        )
        return delta

    def _delta_relocate2(self, sol, u, v, r_src, p_u, r_dst, p_ins) -> Optional[float]:
        """
        [FIX-1] Thêm d[u,v] vào remove_gain.
        Chuỗi bị loại = (prev_u→u→v→next_v), thay bằng (prev_u→next_v).
        Chi phí loại = d[prev_u,u] + d[u,v] + d[v,next_v] - d[prev_u,next_v].
        """
        route_s = sol[r_src]; route_d = sol[r_dst]
        d = self.matrix

        if p_u + 2 >= len(route_s):
            return None
        next_v = route_s[p_u + 2]
        if next_v == 0:
            return None

        prev_u = route_s[p_u - 1]
        # [FIX-1] Đúng: bao gồm d[u,v] trong remove_gain
        remove_gain = (d[prev_u,u] + d[u,v] + d[v,next_v]
                       - d[prev_u,next_v])

        prev_d = route_d[p_ins-1]; next_d = route_d[p_ins]
        insert_cost = d[prev_d,u] + d[u,v] + d[v,next_d] - d[prev_d,next_d]

        delta = insert_cost - remove_gain

        seg_dem = self.demands.get(u,0) + self.demands.get(v,0)
        load_s  = self._route_load(route_s)
        load_d  = self._route_load(route_d)
        delta  += self.lam * (
            max(0.0, load_s - seg_dem - self.capacity) +
            max(0.0, load_d + seg_dem - self.capacity) -
            max(0.0, load_s - self.capacity) -
            max(0.0, load_d - self.capacity)
        )
        return delta

    def _delta_swap(self, sol, u, v, r_u, p_u, r_v, p_v) -> Optional[float]:
        route_u = sol[r_u]; route_v = sol[r_v]
        d = self.matrix
        pu_prev = route_u[p_u-1]; pu_next = route_u[p_u+1]
        pv_prev = route_v[p_v-1]; pv_next = route_v[p_v+1]

        if r_u == r_v:
            if abs(p_u - p_v) == 1: return None
            old = d[pu_prev,u]+d[u,pu_next] + d[pv_prev,v]+d[v,pv_next]
            new = d[pu_prev,v]+d[v,pu_next] + d[pv_prev,u]+d[u,pv_next]
            return float(new - old)
        else:
            old = d[pu_prev,u]+d[u,pu_next] + d[pv_prev,v]+d[v,pv_next]
            new = d[pu_prev,v]+d[v,pu_next] + d[pv_prev,u]+d[u,pv_next]
            delta = float(new - old)
            u_dem = self.demands.get(u,0); v_dem = self.demands.get(v,0)
            load_u = self._route_load(route_u); load_v = self._route_load(route_v)
            delta += self.lam * (
                max(0.0, load_u - u_dem + v_dem - self.capacity) +
                max(0.0, load_v - v_dem + u_dem - self.capacity) -
                max(0.0, load_u - self.capacity) -
                max(0.0, load_v - self.capacity)
            )
            return delta

    def _delta_2opt_star(self, sol, r1, i, r2, j) -> Optional[float]:
        route1 = sol[r1]; route2 = sol[r2]
        d = self.matrix
        if i == 0 or i >= len(route1)-1: return None
        if j == 0 or j >= len(route2)-1: return None

        A = route1[i]; C = route1[i+1]
        B = route2[j]; D = route2[j+1]
        delta = float(d[A,D] + d[B,C] - d[A,C] - d[B,D])

        tail1_load = sum(self.demands.get(x,0) for x in route1[i+1:] if x!=0)
        tail2_load = sum(self.demands.get(x,0) for x in route2[j+1:] if x!=0)
        head1_load = self._route_load(route1) - tail1_load
        head2_load = self._route_load(route2) - tail2_load
        delta += self.lam * (
            max(0.0, head1_load + tail2_load - self.capacity) +
            max(0.0, head2_load + tail1_load - self.capacity) -
            max(0.0, self._route_load(route1) - self.capacity) -
            max(0.0, self._route_load(route2) - self.capacity)
        )
        return delta

    # ── Apply moves ────────────────────────────────────────────────────

    def _apply_relocate(self, sol, u, r_src, p_u, r_dst, p_ins):
        """[FIX-4] Không gọi sol[:]=filter ở đây; caller tự rebuild."""
        sol[r_src].pop(p_u)
        sol[r_dst].insert(p_ins, u)

    def _apply_relocate2(self, sol, u, v, r_src, p_u, r_dst, p_ins):
        """[FIX-4] Không filter. Xóa v trước (index p_u+1), rồi u (p_u)."""
        sol[r_src].pop(p_u + 1)
        sol[r_src].pop(p_u)
        sol[r_dst].insert(p_ins, v)
        sol[r_dst].insert(p_ins, u)

    def _apply_swap(self, sol, u, v, r_u, p_u, r_v, p_v):
        sol[r_u][p_u] = v
        sol[r_v][p_v] = u

    def _apply_2opt_star(self, sol, r1, i, r2, j):
        """[FIX-4] Không filter; caller rebuild pos_map từ đầu."""
        new_r1 = sol[r1][:i+1] + sol[r2][j+1:]
        new_r2 = sol[r2][:j+1] + sol[r1][i+1:]
        sol[r1] = new_r1
        sol[r2] = new_r2

    # ── Perturbation ───────────────────────────────────────────────────

    def _double_bridge(self, sol: Solution) -> Solution:
        """[FIX-3] Không recursive; giới hạn k = min(8, len//2)."""
        new_sol = self._copy_sol(sol)
        # [FIX-3] Tính k an toàn, không gây infinite recursion
        k = min(8, len(new_sol) // 2)
        if k < 2:
            return new_sol  # không đủ xe để perturb

        idx_to_shake = random.sample(range(len(new_sol)), k)
        flat = []
        for idx in sorted(idx_to_shake, reverse=True):
            route = new_sol.pop(idx)
            flat.extend([n for n in route if n != 0])

        if len(flat) < 4:
            return self._copy_sol(sol)

        positions = sorted(random.sample(range(1, len(flat)), min(4, len(flat)-1)))
        while len(positions) < 4:
            positions.append(len(flat))
        a, b, c, _ = positions[:4]

        seg0 = flat[:a]; seg1 = flat[a:b]; seg2 = flat[b:c]; seg3 = flat[c:]
        new_flat = seg0 + seg2 + seg1 + seg3

        shaken = self._rebuild_from_flat(new_flat)
        new_sol.extend(shaken)
        return new_sol

    def _rebuild_from_flat(self, flat: List[int]) -> Solution:
        sol = []; curr = [0]; load = 0.0
        for node in flat:
            d = self.demands.get(node, 0)
            if load + d > self.capacity:
                curr.append(0); sol.append(curr)
                curr = [0]; load = 0.0
            curr.append(node); load += d
        curr.append(0); sol.append(curr)
        while len(sol) > self.max_v:
            last = sol.pop()
            sol[-1] = sol[-1][:-1] + last[1:]
        return sol

    # ── Post-optimization ──────────────────────────────────────────────

    def _intra_or_opt(self, sol: Solution) -> Solution:
        d = self.matrix
        for r in sol:
            if len(r) <= 4: continue
            max_passes = 30; pass_count = 0; improved = True
            while improved and pass_count < max_passes:
                improved = False; pass_count += 1; n_r = len(r)
                for i in range(1, n_r-1):
                    node = r[i]; prev_i = r[i-1]; next_i = r[i+1]
                    gain_remove = d[prev_i,node]+d[node,next_i]-d[prev_i,next_i]
                    best_gain = 1e-6; best_j = -1
                    for j in range(1, n_r-1):
                        if j == i or j == i-1: continue
                        prev_j = r[j-1]; next_j = r[j]
                        gain_insert = d[prev_j,node]+d[node,next_j]-d[prev_j,next_j]
                        gain = gain_remove - gain_insert
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
        curr_sol  = self._copy_sol(initial_solution)
        self._clean_empty_routes(curr_sol)
        best_sol  = self._copy_sol(curr_sol)
        best_dist = self._total_cost(curr_sol)

        tabu_dict: Dict[tuple, int] = {}
        no_improve = 0; iteration = 0
        infeasible_count = 0

        print(f"[GTS] Bắt đầu: {len(curr_sol)} xe | {best_dist/100:.2f} km | λ={self.lam:.1f}")

        while iteration < self.max_iter:
            if no_improve >= self.max_no_improve:
                print(f"[GTS] Dừng iter {iteration}: {no_improve} vòng không cải thiện")
                break

            # Diversification
            if no_improve == self.max_no_improve // 2:
                perturbed = self._double_bridge(self._copy_sol(best_sol))
                self._clean_empty_routes(perturbed)
                p_cost = self._total_cost(perturbed)
                if p_cost < best_dist:
                    best_dist = p_cost; best_sol = self._copy_sol(perturbed)
                    no_improve = 0
                    print(f"  [GTS] Perturbation NEW BEST: {best_dist/100:.2f} km")
                if p_cost < best_dist * 1.1:
                    curr_sol = perturbed

            # [FIX-4] Rebuild pos_map từ đầu mỗi iteration (sau clean)
            self._clean_empty_routes(curr_sol)
            pos_map   = self._node_positions(curr_sol)
            base_cost = self._penalized_cost(curr_sol)

            best_move_delta = float('inf')
            best_move = None; best_move_key = None; best_move_type = None

            for u in list(pos_map.keys()):
                r_u, p_u = pos_map[u]
                route_u  = curr_sol[r_u]

                # ── Or-opt-1 ────────────────────────────────────────
                for v in self._granular_neighbors.get(u, []):
                    if v not in pos_map: continue
                    r_v, p_v = pos_map[v]
                    if r_v == r_u: continue

                    for p_ins in [p_v, p_v+1]:
                        if p_ins < 1 or p_ins >= len(curr_sol[r_v]): continue
                        delta = self._delta_relocate(curr_sol, u, r_u, p_u, r_v, p_ins)
                        if delta is None: continue
                        key = ('R1', u, r_v)
                        in_tabu = tabu_dict.get(key, -1) >= iteration
                        asp = (base_cost + delta < best_dist - 1e-6)
                        if (not in_tabu or asp) and delta < best_move_delta:
                            best_move_delta = delta
                            best_move = (u, r_u, p_u, r_v, p_ins)
                            best_move_key = key; best_move_type = 'rel1'
                    # chỉ xét 2 vị trí chèn tốt nhất, break ngay
                    if best_move_type == 'rel1' and best_move and best_move[0] == u:
                        break

                # ── Or-opt-2 ────────────────────────────────────────
                if p_u + 1 < len(route_u) - 1:
                    v_next = route_u[p_u + 1]
                    if v_next != 0:
                        for nb in self._granular_neighbors.get(u, [])[:10]:
                            if nb not in pos_map: continue
                            r_nb, p_nb = pos_map[nb]
                            if r_nb == r_u: continue
                            for p_ins in range(1, len(curr_sol[r_nb])):
                                delta2 = self._delta_relocate2(
                                    curr_sol, u, v_next, r_u, p_u, r_nb, p_ins)
                                if delta2 is None: continue
                                key2 = ('R2', u, r_nb)
                                in_tabu2 = tabu_dict.get(key2, -1) >= iteration
                                asp2 = (base_cost + delta2 < best_dist - 1e-6)
                                if (not in_tabu2 or asp2) and delta2 < best_move_delta:
                                    best_move_delta = delta2
                                    best_move = (u, v_next, r_u, p_u, r_nb, p_ins)
                                    best_move_key = key2; best_move_type = 'rel2'
                                break

                # ── SWAP ────────────────────────────────────────────
                for v in self._granular_neighbors.get(u, []):
                    if v not in pos_map: continue
                    r_v, p_v = pos_map[v]
                    if r_u == r_v and p_u >= p_v: continue
                    ds = self._delta_swap(curr_sol, u, v, r_u, p_u, r_v, p_v)
                    if ds is None: continue
                    keys = ('SW', min(u,v), max(u,v))
                    in_tabu_s = tabu_dict.get(keys, -1) >= iteration
                    asp_s = (base_cost + ds < best_dist - 1e-6)
                    if (not in_tabu_s or asp_s) and ds < best_move_delta:
                        best_move_delta = ds
                        best_move = (u, v, r_u, p_u, r_v, p_v)
                        best_move_key = keys; best_move_type = 'swap'

                # ── 2-opt* ──────────────────────────────────────────
                for v in self._granular_neighbors.get(u, [])[:8]:
                    if v not in pos_map: continue
                    r_v, p_v = pos_map[v]
                    if r_v == r_u: continue
                    d2s = self._delta_2opt_star(curr_sol, r_u, p_u, r_v, p_v)
                    if d2s is None: continue
                    key2s = ('2S', r_u, p_u, r_v, p_v)
                    in_tabu_2s = tabu_dict.get(key2s, -1) >= iteration
                    asp_2s = (base_cost + d2s < best_dist - 1e-6)
                    if (not in_tabu_2s or asp_2s) and d2s < best_move_delta:
                        best_move_delta = d2s
                        best_move = (r_u, p_u, r_v, p_v)
                        best_move_key = key2s; best_move_type = '2opts'

            # Apply best move
            if best_move is None:
                no_improve += 1; iteration += 1; continue

            new_sol = self._copy_sol(curr_sol)
            if best_move_type == 'rel1':
                u, r_src, p_u, r_dst, p_ins = best_move
                self._apply_relocate(new_sol, u, r_src, p_u, r_dst, p_ins)
            elif best_move_type == 'rel2':
                u, v_nxt, r_src, p_u, r_dst, p_ins = best_move
                self._apply_relocate2(new_sol, u, v_nxt, r_src, p_u, r_dst, p_ins)
            elif best_move_type == 'swap':
                u, v, r_u, p_u, r_v, p_v = best_move
                self._apply_swap(new_sol, u, v, r_u, p_u, r_v, p_v)
            elif best_move_type == '2opts':
                r1, i, r2, j = best_move
                self._apply_2opt_star(new_sol, r1, i, r2, j)

            # [FIX-4] Clean AFTER apply, trước iteration tiếp theo
            self._clean_empty_routes(new_sol)
            curr_sol = new_sol

            # Tabu update
            tenure = random.randint(self.tau_min, self.tau_max)
            tabu_dict[best_move_key] = iteration + tenure
            if iteration % 50 == 0:
                tabu_dict = {k: v for k, v in tabu_dict.items() if v >= iteration}

            # Check improvement
            curr_cost = self._total_cost(curr_sol)
            is_feasible = all(self._route_load(r) <= self.capacity for r in curr_sol)

            if is_feasible and curr_cost < best_dist:
                best_dist = curr_cost; best_sol = self._copy_sol(curr_sol)
                no_improve = 0
                print(f"  [GTS iter {iteration:5d}] ✓ {curr_cost/100:.2f} km "
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
                print(f"  [GTS iter {iteration:5d}] best={best_dist/100:.2f} km "
                      f"| NoImprove={no_improve}/{self.max_no_improve} | λ={self.lam:.1f}")

            iteration += 1

        print(f"\n[GTS] Post-opt Or-opt intra-route...")
        best_sol  = self._intra_or_opt(best_sol)
        best_dist = self._total_cost(best_sol)
        print(f"[GTS] Hoàn tất: {best_dist/100:.2f} km | {len(best_sol)} xe")
        return best_sol, best_dist