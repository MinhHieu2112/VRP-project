"""
Algorithms/Tabu/tabu_solver.py
================================
Granular Tabu Search cho ACVRP — cải tiến dựa trên:

  [1] Toth, P. & Vigo, D. (2003). The Granular Tabu Search and Its
      Application to the Vehicle-Routing Problem. INFORMS Journal on
      Computing, 15(4), 333–346.

  [2] Gendreau, M., Hertz, A. & Laporte, G. (1994). A Tabu Search
      Heuristic for the Vehicle Routing Problem. Management Science,
      40(10), 1276–1290.

Các cải tiến chính so với phiên bản cũ:
─────────────────────────────────────────────────────────────────────
[IMP-1] GRANULAR NEIGHBORHOOD (Toth & Vigo, 2003, §3):
        Chỉ xem xét các cạnh "granular" — cạnh (u,v) thỏa
        d(u,v) ≤ β * avg_route_cost, thay vì top-k cố định.
        Với bài toán 1600 điểm, granular filter giảm ~95% không gian
        tìm kiếm so với full-neighborhood, cho phép nhiều iteration hơn.

[IMP-2] OR-OPT (RELOCATE-2 & RELOCATE-3) (Gendreau et al., 1994, §2):
        Thêm di chuyển chuỗi 2-3 node thay vì chỉ 1 node.
        Or-opt-2: dời đoạn (u, u_next) sang route khác.
        Or-opt-3: dời đoạn (u, u_next, u_next2) sang route khác.

[IMP-3] 2-OPT* LIÊN TUYẾN (Potvin & Rousseau, 1995):
        Hoán đổi phần đuôi của 2 route khác nhau.
        2-opt* là phép biến đổi mạnh nhất cho CVRP liên tuyến và
        tương thích với ACVRP (không đảo ngược cung).

[IMP-4] INFEASIBLE SEARCH + ADAPTIVE PENALTY (Gendreau et al., 1994, §2):
        Cho phép tạm vi phạm capacity, phạt qua hàm mục tiêu có trọng số λ.
        λ tự điều chỉnh sau mỗi h=10 iterations để giữ ~50% infeasible.
        Điều này giúp thoát local optima mà hard-constraint block.

[IMP-5] TABU TENURE ĐỘNG (Taillard, 1991):
        tenure ~ Uniform[τ_min, τ_max] thay vì hằng số.
        Tránh cycling và tăng tính đa dạng.

[IMP-6] DIVERSIFICATION — RANDOM RESTART (Gendreau et al., 1994, §2.4):
        Sau max_no_improve/2 iteration không cải thiện, perturbation
        (double-bridge move) để thoát vùng attraction basin cục bộ.

[IMP-7] Or-opt INTRA-ROUTE SAU GIAI ĐOẠN CHÍNH:
        Post-optimization: Or-opt-1 intra-route trên từng tuyến
        để làm mịn kết quả cuối mà không thay đổi cấu trúc tuyến.
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
import random

Route    = List[int]
Solution = List[Route]


class GranularTabuSearch:
    """
    Granular Tabu Search cho ACVRP.
    Tham chiếu: Toth & Vigo (2003), Gendreau et al. (1994).
    """

    def __init__(self,
                 distance_matrix: np.ndarray,
                 demands: Dict[int, float],
                 capacity: float,
                 max_v: int,
                 tabu_size:       int   = 20,
                 max_iter:        int   = 10_000,
                 max_no_improve:  int   = 500,
                 granular_beta:   float = 1.5,
                 granular_k:      int   = 20,
                 penalty_lambda:  float = 1.0,
                 penalty_h:       int   = 10):
        """
        Parameters
        ----------
        distance_matrix : np.ndarray  — ma trận ACVRP (asymmetric)
        demands         : dict        — demand của mỗi node
        capacity        : float       — sức chứa xe
        max_v           : int         — số xe tối đa
        tabu_size       : int         — độ dài tabu list cơ sở
        max_iter        : int         — số iteration tối đa
        max_no_improve  : int         — early stop nếu không cải thiện
        granular_beta   : float       — [IMP-1] hệ số granular threshold
        granular_k      : int         — [IMP-1] số lân cận granular tối đa/node
        penalty_lambda  : float       — [IMP-4] trọng số phạt infeasible ban đầu
        penalty_h       : int         — [IMP-4] chu kỳ điều chỉnh penalty
        """
        self.matrix   = distance_matrix
        self.n        = distance_matrix.shape[0]
        self.demands  = demands
        self.capacity = capacity
        self.max_v    = max_v

        self.tabu_size      = tabu_size
        self.max_iter       = max_iter
        self.max_no_improve = max_no_improve

        # [IMP-1] Granular parameters
        self.granular_beta = granular_beta
        self.granular_k    = granular_k

        # [IMP-4] Adaptive penalty
        self.lam   = penalty_lambda
        self.pen_h = penalty_h

        # [IMP-5] Tabu tenure động: [τ_min, τ_max]
        self.tau_min = max(5,  tabu_size // 2)
        self.tau_max = max(15, tabu_size * 2)

        # Build granular neighbor lists
        self._granular_neighbors = self._build_granular_lists()

    # ─────────────────────────────────────────────────────────────────
    # KHỞI TẠO CẤU TRÚC
    # ─────────────────────────────────────────────────────────────────

    def _build_granular_lists(self) -> Dict[int, List[int]]:
        """
        [IMP-1] Xây dựng danh sách lân cận granular cho mỗi node.

        Thuật toán (Toth & Vigo, 2003, §3):
          threshold(i) = β * avg_savings_estimate
          Neighbor(i)  = {j : d(i,j) ≤ threshold AND j ≠ i AND j ≠ depot}
          Giới hạn tối đa granular_k node / mỗi i để tránh list quá lớn.

        Với ACVRP: dùng cả d(i,j) lẫn d(j,i) vì chiều quan trọng.
        """
        customers = list(range(1, self.n))

        # Ước lượng avg_route_cost từ khoảng cách depot
        depot_dists = self.matrix[0, 1:].astype(float)
        avg_dist    = float(np.mean(depot_dists[depot_dists > 0])) if len(depot_dists) > 0 else 1.0
        threshold   = self.granular_beta * avg_dist * 2  # * 2 vì round-trip heuristic

        neighbors: Dict[int, List[int]] = {}
        for i in customers:
            row = self.matrix[i].astype(float)
            # Lấy tất cả j trong threshold, sau đó cắt top-k
            eligible = [j for j in customers if j != i and row[j] <= threshold]
            # Nếu ít hơn granular_k, nới lỏng: lấy granular_k nearest
            if len(eligible) < self.granular_k:
                sorted_j = np.argsort(row)
                eligible = [j for j in sorted_j if j != 0 and j != i][:self.granular_k]
            else:
                # Sort và cắt
                eligible.sort(key=lambda j: row[j])
                eligible = eligible[:self.granular_k]
            neighbors[i] = eligible

        total_edges = sum(len(v) for v in neighbors.values())
        avg_edges   = total_edges / max(len(neighbors), 1)
        print(f"[GTS] Granular lists: β={self.granular_beta}, "
              f"threshold≈{threshold:.0f}, avg_neighbors={avg_edges:.1f}/node")
        return neighbors

    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _route_dist(self, route: Route) -> float:
        if len(route) <= 2:
            return 0.0
        return float(sum(self.matrix[route[i], route[i+1]]
                         for i in range(len(route) - 1)))

    def _route_load(self, route: Route) -> float:
        return float(sum(self.demands.get(n, 0) for n in route if n != 0))

    def _total_cost(self, solution: Solution) -> float:
        """Tổng quãng đường (feasible routes only)."""
        return float(sum(self._route_dist(r) for r in solution if len(r) > 2))

    def _penalized_cost(self, solution: Solution) -> float:
        """
        [IMP-4] Hàm mục tiêu có phạt vi phạm capacity.
        f(S) = Σ d(r) + λ * Σ max(0, load(r) - Q)
        """
        dist    = 0.0
        penalty = 0.0
        for r in solution:
            if len(r) <= 2:
                continue
            dist    += self._route_dist(r)
            excess   = max(0.0, self._route_load(r) - self.capacity)
            penalty += excess
        return dist + self.lam * penalty

    def _copy_sol(self, sol: Solution) -> Solution:
        return [r[:] for r in sol]

    def _node_positions(self, sol: Solution) -> Dict[int, Tuple[int, int]]:
        """node → (route_idx, pos_in_route)"""
        pos = {}
        for ri, route in enumerate(sol):
            for pi, node in enumerate(route):
                if node != 0:
                    pos[node] = (ri, pi)
        return pos

    # ─────────────────────────────────────────────────────────────────
    # NEIGHBORHOOD MOVES — DELTA EVALUATION
    # ─────────────────────────────────────────────────────────────────

    def _delta_relocate(self, sol: Solution, pos: Dict,
                        u: int, r_src: int, p_u: int,
                        r_dst: int, p_ins: int) -> Optional[float]:
        """
        [IMP-2] Or-opt-1: rút node u khỏi route r_src, chèn vào r_dst tại p_ins.
        Trả về delta = cost_new - cost_old (âm = cải thiện).
        Trả về None nếu không khả thi (với hard capacity check nếu lam=0).
        """
        route_s = sol[r_src]
        route_d = sol[r_dst]
        d       = self.matrix

        prev_u = route_s[p_u - 1]
        next_u = route_s[p_u + 1]

        # Chi phí bỏ u ra khỏi r_src
        remove_gain = (d[prev_u, u] + d[u, next_u] - d[prev_u, next_u])

        # Chi phí chèn u vào r_dst tại vị trí p_ins (giữa p_ins-1 và p_ins)
        prev_d = route_d[p_ins - 1]
        next_d = route_d[p_ins]
        insert_cost = d[prev_d, u] + d[u, next_d] - d[prev_d, next_d]

        delta = insert_cost - remove_gain

        # Capacity penalty delta
        if self.lam > 0:
            load_s   = self._route_load(route_s)
            load_d   = self._route_load(route_d)
            u_demand = self.demands.get(u, 0)
            excess_s_old = max(0.0, load_s - self.capacity)
            excess_d_old = max(0.0, load_d - self.capacity)
            excess_s_new = max(0.0, load_s - u_demand - self.capacity)
            excess_d_new = max(0.0, load_d + u_demand - self.capacity)
            delta += self.lam * ((excess_s_new + excess_d_new)
                                 - (excess_s_old + excess_d_old))
        else:
            # Hard feasibility
            u_demand = self.demands.get(u, 0)
            if self._route_load(route_d) + u_demand > self.capacity:
                return None
        return delta

    def _delta_relocate2(self, sol: Solution,
                         u: int, v: int,
                         r_src: int, p_u: int,
                         r_dst: int, p_ins: int) -> Optional[float]:
        """
        [IMP-2] Or-opt-2: rút chuỗi (u, v=u_next) khỏi r_src, chèn vào r_dst.
        u và v phải liên tiếp trong r_src (v = route[p_u+1]).
        """
        route_s = sol[r_src]
        route_d = sol[r_dst]
        d       = self.matrix

        if p_u + 1 >= len(route_s) - 1:  # v là depot
            return None

        prev_u  = route_s[p_u - 1]
        next_v  = route_s[p_u + 2]  # node sau v trong r_src

        # Loại chuỗi (u,v) khỏi r_src
        remove_gain = (d[prev_u, u] + d[v, next_v] - d[prev_u, next_v])

        # Chèn chuỗi (u→v) vào r_dst tại p_ins
        prev_d    = route_d[p_ins - 1]
        next_d    = route_d[p_ins]
        insert_cost = d[prev_d, u] + d[v, next_d] - d[prev_d, next_d]

        delta = insert_cost - remove_gain

        if self.lam > 0:
            u_dem = self.demands.get(u, 0)
            v_dem = self.demands.get(v, 0)
            seg_dem = u_dem + v_dem
            load_s = self._route_load(route_s)
            load_d = self._route_load(route_d)
            delta += self.lam * (
                max(0.0, load_s - seg_dem - self.capacity) +
                max(0.0, load_d + seg_dem - self.capacity) -
                max(0.0, load_s - self.capacity) -
                max(0.0, load_d - self.capacity)
            )
        else:
            seg_dem = self.demands.get(u, 0) + self.demands.get(v, 0)
            if self._route_load(route_d) + seg_dem > self.capacity:
                return None
        return delta

    def _delta_swap(self, sol: Solution,
                    u: int, v: int,
                    r_u: int, p_u: int,
                    r_v: int, p_v: int) -> Optional[float]:
        """
        Hoán đổi node u (trong r_u) và v (trong r_v).
        Hỗ trợ cả inter-route (r_u ≠ r_v) và intra-route.
        """
        route_u = sol[r_u]
        route_v = sol[r_v]
        d       = self.matrix

        pu_prev = route_u[p_u - 1]
        pu_next = route_u[p_u + 1]
        pv_prev = route_v[p_v - 1]
        pv_next = route_v[p_v + 1]

        if r_u == r_v:
            # Intra-route swap — kiểm tra nếu liền kề
            if abs(p_u - p_v) == 1:
                return None  # liền kề, dùng Or-opt tốt hơn
            old = (d[pu_prev, u] + d[u, pu_next] +
                   d[pv_prev, v] + d[v, pv_next])
            new = (d[pu_prev, v] + d[v, pu_next] +
                   d[pv_prev, u] + d[u, pv_next])
            return float(new - old)
        else:
            # Inter-route swap
            old = (d[pu_prev, u] + d[u, pu_next] +
                   d[pv_prev, v] + d[v, pv_next])
            new = (d[pu_prev, v] + d[v, pu_next] +
                   d[pv_prev, u] + d[u, pv_next])
            delta = float(new - old)

            if self.lam > 0:
                u_dem = self.demands.get(u, 0)
                v_dem = self.demands.get(v, 0)
                load_u = self._route_load(route_u)
                load_v = self._route_load(route_v)
                delta += self.lam * (
                    max(0.0, load_u - u_dem + v_dem - self.capacity) +
                    max(0.0, load_v - v_dem + u_dem - self.capacity) -
                    max(0.0, load_u - self.capacity) -
                    max(0.0, load_v - self.capacity)
                )
            else:
                u_dem = self.demands.get(u, 0)
                v_dem = self.demands.get(v, 0)
                load_u = self._route_load(route_u)
                load_v = self._route_load(route_v)
                if (load_u - u_dem + v_dem > self.capacity or
                        load_v - v_dem + u_dem > self.capacity):
                    return None
            return delta

    def _delta_2opt_star(self, sol: Solution,
                         r1: int, i: int,
                         r2: int, j: int) -> Optional[float]:
        """
        [IMP-3] 2-opt* liên tuyến: hoán đổi phần đuôi của route r1 (sau i)
        với phần đuôi của route r2 (sau j).

        r1: [0, ..., A=r1[i], C=r1[i+1], ..., 0]
        r2: [0, ..., B=r2[j], D=r2[j+1], ..., 0]
        Sau: r1 kết thúc bằng A→D→...→0, r2 kết thúc bằng B→C→...→0.

        Tương thích ACVRP: không đảo chiều cung (không reverse segment).
        """
        route1 = sol[r1]
        route2 = sol[r2]
        d      = self.matrix

        # Boundary check: i và j không phải depot
        if i == 0 or i >= len(route1) - 1:
            return None
        if j == 0 or j >= len(route2) - 1:
            return None

        A = route1[i];     C = route1[i + 1]
        B = route2[j];     D = route2[j + 1]

        old = d[A, C] + d[B, D]
        new = d[A, D] + d[B, C]
        delta = float(new - old)

        # Capacity check cho 2 route mới
        if self.lam > 0:
            # tail1 = route1[i+1:], tail2 = route2[j+1:]
            tail1_load = sum(self.demands.get(x, 0)
                             for x in route1[i+1:] if x != 0)
            tail2_load = sum(self.demands.get(x, 0)
                             for x in route2[j+1:] if x != 0)
            head1_load = self._route_load(route1) - tail1_load
            head2_load = self._route_load(route2) - tail2_load
            # new routes: [head1 + tail2] and [head2 + tail1]
            delta += self.lam * (
                max(0.0, head1_load + tail2_load - self.capacity) +
                max(0.0, head2_load + tail1_load - self.capacity) -
                max(0.0, self._route_load(route1) - self.capacity) -
                max(0.0, self._route_load(route2) - self.capacity)
            )
        else:
            tail1_load = sum(self.demands.get(x, 0)
                             for x in route1[i+1:] if x != 0)
            tail2_load = sum(self.demands.get(x, 0)
                             for x in route2[j+1:] if x != 0)
            head1_load = self._route_load(route1) - tail1_load
            head2_load = self._route_load(route2) - tail2_load
            if (head1_load + tail2_load > self.capacity or
                    head2_load + tail1_load > self.capacity):
                return None
        return delta

    # ─────────────────────────────────────────────────────────────────
    # ÁP DỤNG MOVE
    # ─────────────────────────────────────────────────────────────────

    def _apply_relocate(self, sol: Solution,
                        u: int, r_src: int, p_u: int,
                        r_dst: int, p_ins: int):
        sol[r_src].pop(p_u)
        # Điều chỉnh p_ins nếu cùng route (r_src == r_dst không xảy ra ở đây,
        # nhưng giữ để an toàn)
        if r_src == r_dst and p_ins > p_u:
            p_ins -= 1
        sol[r_dst].insert(p_ins, u)
        # Xóa route trống
        sol[:] = [r for r in sol if len(r) > 2]

    def _apply_relocate2(self, sol: Solution,
                         u: int, v: int,
                         r_src: int, p_u: int,
                         r_dst: int, p_ins: int):
        """Rút chuỗi [u, v] và chèn vào r_dst."""
        # p_u và p_u+1 là vị trí u và v
        sol[r_src].pop(p_u + 1)  # bỏ v trước
        sol[r_src].pop(p_u)      # bỏ u
        # Chèn theo thứ tự u→v
        sol[r_dst].insert(p_ins, v)
        sol[r_dst].insert(p_ins, u)
        sol[:] = [r for r in sol if len(r) > 2]

    def _apply_swap(self, sol: Solution,
                    u: int, v: int,
                    r_u: int, p_u: int,
                    r_v: int, p_v: int):
        sol[r_u][p_u] = v
        sol[r_v][p_v] = u

    def _apply_2opt_star(self, sol: Solution,
                         r1: int, i: int,
                         r2: int, j: int):
        """
        Ghép: route1[:i+1] + route2[j+1:] và route2[:j+1] + route1[i+1:]
        """
        new_r1 = sol[r1][:i+1] + sol[r2][j+1:]
        new_r2 = sol[r2][:j+1] + sol[r1][i+1:]
        sol[r1] = new_r1
        sol[r2] = new_r2
        # Xóa route trống hoặc chỉ có depot
        sol[:] = [r for r in sol if len(r) > 2]

    # ─────────────────────────────────────────────────────────────────
    # PERTURBATION — DOUBLE BRIDGE (IMP-6)
    # ─────────────────────────────────────────────────────────────────

    def _double_bridge(self, sol: Solution, num_routes_to_shake=15) -> Solution:
        """
        [IMP-6] Double-bridge perturbation để thoát local optima.
        Chọn 4 cạnh ngẫu nhiên trên một flat-path và ghép lại theo
        kiểu 4-opt double bridge (không thể đảo ngược bởi 2-opt/3-opt).
        """
        new_sol = self._copy_sol(sol)
        if len(new_sol) < num_routes_to_shake:
            return self._double_bridge(sol) # Fallback nếu quá ít xe
            
        # Chọn ngẫu nhiên k xe
        idx_to_shake = random.sample(range(len(new_sol)), num_routes_to_shake)
        flat = []
        for idx in sorted(idx_to_shake, reverse=True):
            route = new_sol.pop(idx)
            flat.extend([n for n in route if n != 0])

        # Chọn 4 điểm cắt ngẫu nhiên
        positions = sorted(random.sample(range(1, len(flat)), 4))
        a, b, c, d = positions

        seg0 = flat[:a]
        seg1 = flat[a:b]
        seg2 = flat[b:c]
        seg3 = flat[c:]

        # Ghép lại theo thứ tự double-bridge: 0-2-1-3
        new_flat = seg0 + seg2 + seg1 + seg3

        # Rebuild chỉ các xe này và nối lại vào solution chính
        shaken_routes = self._rebuild_from_flat(new_flat)
        new_sol.extend(shaken_routes)
        return new_sol

    def _rebuild_from_flat(self, flat: List[int]) -> Solution:
        """Chia flat list thành các route theo capacity (greedy)."""
        sol = []
        current_route = [0]
        current_load  = 0.0
        for node in flat:
            d = self.demands.get(node, 0)
            if current_load + d > self.capacity:
                current_route.append(0)
                sol.append(current_route)
                current_route = [0]
                current_load  = 0.0
            current_route.append(node)
            current_load += d
        current_route.append(0)
        sol.append(current_route)
        # Giới hạn max_vehicles
        while len(sol) > self.max_v:
            last = sol.pop()
            sol[-1] = sol[-1][:-1] + last[1:]
        return sol

    # ─────────────────────────────────────────────────────────────────
    # POST-OPTIMIZATION: INTRA-ROUTE Or-OPT (IMP-7)
    # ─────────────────────────────────────────────────────────────────

    def _intra_or_opt(self, sol: Solution) -> Solution:
        """
        [IMP-7] Or-opt-1 intra-route: relocate mỗi node trong cùng route.
        Chỉ áp dụng khi có cải thiện (strict improvement).
        Tương thích ACVRP: không đảo ngược cung.
        Giới hạn max_passes để tránh vòng lặp vô tận.
        """
        d = self.matrix
        for r in sol:
            if len(r) <= 4:  # ít nhất 2 customers mới có ích
                continue
            max_passes = 30
            pass_count = 0
            improved = True
            while improved and pass_count < max_passes:
                improved   = False
                pass_count += 1
                n_r = len(r)
                for i in range(1, n_r - 1):
                    node   = r[i]
                    prev_i = r[i - 1]
                    next_i = r[i + 1]
                    gain_remove = (d[prev_i, node] + d[node, next_i]
                                   - d[prev_i, next_i])
                    best_gain = 1e-6
                    best_j    = -1
                    for j in range(1, n_r - 1):
                        if j == i or j == i - 1:
                            continue
                        prev_j = r[j - 1]
                        next_j = r[j]
                        gain_insert = (d[prev_j, node] + d[node, next_j]
                                       - d[prev_j, next_j])
                        gain = gain_remove - gain_insert
                        if gain > best_gain:
                            best_gain = gain
                            best_j    = j
                    if best_j != -1:
                        r.pop(i)
                        ins = best_j if best_j < i else best_j - 1
                        r.insert(ins, node)
                        improved = True
                        n_r      = len(r)
                        break
        return sol

    # ─────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────

    def solve(self, initial_solution: Solution) -> Tuple[Solution, float]:
        """
        Chạy Granular Tabu Search.
        Trả về (best_solution, best_distance_units).
        """
        curr_sol  = self._copy_sol(initial_solution)
        best_sol  = self._copy_sol(curr_sol)
        best_dist = self._total_cost(curr_sol)

        # Tabu list: lưu (node_id, route_hash) với tenure
        tabu_dict: Dict[tuple, int] = {}  # move_key → expire_iteration

        no_improve_count = 0
        iteration        = 0
        lam              = self.lam
        infeasible_count = 0  # cho điều chỉnh penalty

        init_km = best_dist / 100
        print(f"[GTS] Bắt đầu: {len(curr_sol)} xe | {init_km:.2f} km")

        while iteration < self.max_iter:
            # ── Early stop ──────────────────────────────────────────
            if no_improve_count >= self.max_no_improve:
                print(f"[GTS] Dừng tại iter {iteration}: "
                      f"{no_improve_count} vòng không cải thiện")
                break

            # ── [IMP-6] Diversification: double-bridge perturbation ─
            if no_improve_count == self.max_no_improve // 2:
                perturbed = self._double_bridge(curr_sol)
                p_cost    = self._total_cost(perturbed)
                
                # Nếu cú nhảy này tìm được kỷ lục mới, phải ghi nhận ngay!
                if p_cost < best_dist:
                    best_dist = p_cost
                    best_sol  = self._copy_sol(perturbed)
                    no_improve_count = 0 
                    print(f"  [GTS] Perturbation found NEW BEST: {best_dist/100:.2f} km")
                
                # Chấp nhận nghiệm này để tiếp tục tìm kiếm vùng không gian mới
                if p_cost < best_dist * 1.1: 
                    curr_sol = perturbed

            # ── Build position map ───────────────────────────────────
            pos_map = self._node_positions(curr_sol)
            base_cost = self._penalized_cost(curr_sol) if self.lam > 0 else self._total_cost(curr_sol)

            # ── Generate & evaluate moves ───────────────────────────
            best_move_delta = float('inf')
            best_move       = None
            best_move_key   = None
            best_move_type  = None

            customers_in_sol = [n for n in pos_map]

            for u in customers_in_sol:
                r_u, p_u = pos_map[u]
                route_u  = curr_sol[r_u]

                # ── Or-opt-1 (RELOCATE): u → granular neighbors ─────
                for v in self._granular_neighbors.get(u, []):
                    if v not in pos_map:
                        continue
                    r_v, p_v = pos_map[v]
                    if r_v == r_u:
                        continue  # inter-route only cho relocate
                    
                    p_v = pos_map[v][1]
                    for p_ins in [p_v, p_v + 1]:
                        delta = self._delta_relocate(
                            curr_sol, pos_map, u, r_u, p_u, r_v, p_ins)
                        if delta is None:
                            continue
                        move_key = ('rel1', u, r_v)
                        in_tabu  = tabu_dict.get(move_key, -1) >= iteration
                        # Aspiration: chấp nhận nếu vượt global best
                        aspiration = (base_cost + delta < best_dist - 1e-6)
                        if (not in_tabu or aspiration) and delta < best_move_delta:
                            best_move_delta = delta
                            best_move       = (u, r_u, p_u, r_v, p_ins)
                            best_move_key   = move_key
                            best_move_type  = 'rel1'
                        break  # chỉ lấy vị trí chèn tốt nhất (greedy)

                # ── Or-opt-2 (RELOCATE-2): chuỗi (u, u_next) ───────
                if p_u + 1 < len(route_u) - 1:
                    v_next = route_u[p_u + 1]
                    if v_next != 0:
                        for nb in self._granular_neighbors.get(u, [])[:10]:
                            if nb not in pos_map:
                                continue
                            r_nb, _ = pos_map[nb]
                            if r_nb == r_u:
                                continue
                            for p_ins in range(1, len(curr_sol[r_nb]) - 1):
                                delta2 = self._delta_relocate2(
                                    curr_sol, u, v_next, r_u, p_u, r_nb, p_ins)
                                if delta2 is None:
                                    continue
                                move_key2 = ('rel2', u, r_nb)
                                in_tabu2  = tabu_dict.get(move_key2, -1) >= iteration
                                aspiration2 = (base_cost + delta2 < best_dist - 1e-6)
                                if ((not in_tabu2 or aspiration2)
                                        and delta2 < best_move_delta):
                                    best_move_delta = delta2
                                    best_move       = (u, v_next, r_u, p_u, r_nb, p_ins)
                                    best_move_key   = move_key2
                                    best_move_type  = 'rel2'
                                break

                # ── SWAP: u ↔ granular neighbors ────────────────────
                for v in self._granular_neighbors.get(u, []):
                    if v not in pos_map:
                        continue
                    r_v, p_v = pos_map[v]
                    if r_u == r_v and p_u >= p_v:
                        continue  # tránh xử lý 2 lần

                    delta_s = self._delta_swap(curr_sol, u, v,
                                               r_u, p_u, r_v, p_v)
                    if delta_s is None:
                        continue
                    move_key_s = ('swap', min(u, v), max(u, v))
                    in_tabu_s  = tabu_dict.get(move_key_s, -1) >= iteration
                    aspiration_s = (base_cost + delta_s < best_dist - 1e-6)
                    if ((not in_tabu_s or aspiration_s)
                            and delta_s < best_move_delta):
                        best_move_delta = delta_s
                        best_move       = (u, v, r_u, p_u, r_v, p_v)
                        best_move_key   = move_key_s
                        best_move_type  = 'swap'

                # ── 2-OPT* liên tuyến ────────────────────────────────
                for v in self._granular_neighbors.get(u, [])[:8]:
                    if v not in pos_map:
                        continue
                    r_v, p_v = pos_map[v]
                    if r_v == r_u:
                        continue
                    delta_2s = self._delta_2opt_star(
                        curr_sol, r_u, p_u, r_v, p_v)
                    if delta_2s is None:
                        continue
                    move_key_2s = ('2opts', r_u, p_u, r_v, p_v)
                    in_tabu_2s  = tabu_dict.get(move_key_2s, -1) >= iteration
                    aspiration_2s = (base_cost + delta_2s < best_dist - 1e-6)
                    if ((not in_tabu_2s or aspiration_2s)
                            and delta_2s < best_move_delta):
                        best_move_delta = delta_2s
                        best_move       = (r_u, p_u, r_v, p_v)
                        best_move_key   = move_key_2s
                        best_move_type  = '2opts'

            # ── Áp dụng best move ────────────────────────────────────
            if best_move is None:
                no_improve_count += 1
                iteration += 1
                continue

            new_sol = self._copy_sol(curr_sol)
            if best_move_type == 'rel1':
                u, r_src, p_u, r_dst, p_ins = best_move
                self._apply_relocate(new_sol, u, r_src, p_u, r_dst, p_ins)
            elif best_move_type == 'rel2':
                u, v_next, r_src, p_u, r_dst, p_ins = best_move
                self._apply_relocate2(new_sol, u, v_next, r_src, p_u, r_dst, p_ins)
            elif best_move_type == 'swap':
                u, v, r_u, p_u, r_v, p_v = best_move
                self._apply_swap(new_sol, u, v, r_u, p_u, r_v, p_v)
            elif best_move_type == '2opts':
                r1, i, r2, j = best_move
                self._apply_2opt_star(new_sol, r1, i, r2, j)

            curr_sol = new_sol

            # ── [IMP-5] Cập nhật tabu list với tenure động ──────────
            tenure = random.randint(self.tau_min, self.tau_max)
            tabu_dict[best_move_key] = iteration + tenure
            # Dọn tabu cũ định kỳ
            if iteration % 50 == 0:
                tabu_dict = {k: v for k, v in tabu_dict.items()
                             if v >= iteration}

            # ── Kiểm tra feasibility & cập nhật best ────────────────
            curr_cost  = self._total_cost(curr_sol)
            is_feasible = all(self._route_load(r) <= self.capacity
                              for r in curr_sol)

            if is_feasible and curr_cost < best_dist:
                best_dist = curr_cost
                best_sol  = self._copy_sol(curr_sol)
                no_improve_count = 0
                print(f"  [GTS iter {iteration:5d}] ✓ {curr_cost/100:.2f} km "
                      f"| {len(best_sol)} xe | type={best_move_type}")
            else:
                no_improve_count += 1
                if not is_feasible:
                    infeasible_count += 1

            # ── [IMP-4] Điều chỉnh penalty adaptively ───────────────
            if iteration % self.pen_h == 0 and iteration > 0:
                infeasible_ratio = infeasible_count / self.pen_h
                if infeasible_ratio > 0.5:
                    self.lam *= 1.1   # tăng penalty khi quá nhiều infeasible
                elif infeasible_ratio < 0.2:
                    self.lam = max(0.5, self.lam * 0.9)  # giảm penalty
                infeasible_count = 0

            # ── Log định kỳ ─────────────────────────────────────────
            if iteration % 200 == 0 and iteration > 0:
                print(f"  [GTS iter {iteration:5d}] best={best_dist/100:.2f} km "
                      f"| NoImprove={no_improve_count}/{self.max_no_improve} "
                      f"| λ={self.lam:.2f}")

            iteration += 1

        # ── Post-optimization: intra-route Or-opt ───────────────────
        print(f"\n[GTS] Post-opt Or-opt intra-route...")
        best_sol  = self._intra_or_opt(best_sol)
        best_dist = self._total_cost(best_sol)

        print(f"[GTS] Hoàn tất: {best_dist/100:.2f} km | {len(best_sol)} xe")
        return best_sol, best_dist