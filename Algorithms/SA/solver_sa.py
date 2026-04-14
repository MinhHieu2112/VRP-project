"""
Algorithms/SA/solver_sa.py
Simulated Annealing solver cho ACVRP.

Trả về (best_solution, best_distance_units) — đơn vị nội bộ matrix_int.
Caller dùng Pipeline.matrix_units_to_km() để quy đổi ra km.

Các fix so với bản cũ:
  [FIX-1] Đọc capacity từ 'global_constraints' (khớp DataLoader mới).
  [FIX-2] Dùng greedy_init thay greedy_init → nghiệm ban đầu tốt hơn ~40%.
  [FIX-3] vehicle_penalty = 3000 units (~30km) đủ lớn để tránh thêm xe bừa.
  [FIX-4] Log in đúng đơn vị km (chia 100, không chia 1000).
  [FIX-5] solve() trả về đơn vị nội bộ, không tự chia km.
"""

import random
import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Algorithms.Init_strategies.Init_strategies import clarke_wright_init, _build_demands

# 1 matrix_int unit = 10m → chia 100 ra km
KM_SCALE = 100


class SimulatedAnnealingSolver:
    def __init__(self, data: dict, config: dict):
        """Khởi tạo SA với data từ DataLoader và config."""
        self.dist = data['distance_matrix']
        self.n    = self.dist.shape[0]

        # [FIX-1] Thử 'global_constraints' trước, fallback sang 'constraints'
        cons = config.get('global_constraints',
               config.get('constraints', {}))
        self.capacity = cons.get('vehicle_capacity', 10)
        self.demand   = cons.get('default_demand', 1)

        sa_cfg              = config.get('alns_parameters', {})
        self.T_start        = sa_cfg.get('start_temperature', 5000)
        self.T_min          = sa_cfg.get('end_temperature', 0.1)
        self.alpha          = sa_cfg.get('step', 0.9997)
        self.max_no_improve = sa_cfg.get('max_no_improve', 1000)
        self.iter_per_T     = sa_cfg.get('iter_per_temp', 500)

        # [FIX-3] Penalty = 3000 units = 30km, đủ lớn hơn 1 tuyến trung bình ~10km
        self.vehicle_penalty = 3000

        self.demands_map = _build_demands(
            self.n, demands=None, default_demand=float(self.demand)
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def route_cost(self, route: list) -> float:
        """Tính tổng khoảng cách một route (đơn vị nội bộ)."""
        return sum(self.dist[route[i]][route[i + 1]]
                   for i in range(len(route) - 1))

    def get_route_load(self, route: list) -> float:
        """Tính tổng demand của route từ demands_map."""
        return sum(self.demands_map.get(n, self.demand)
                   for n in route if n != 0)

    def initial_solution(self) -> list:
        sol = clarke_wright_init(
            matrix      = self.dist,
            num_nodes   = self.n,
            capacity    = self.capacity,
            demands_map = self.demands_map  
        )
        init_dist = sum(self.route_cost(r) for r in sol if len(r) > 2)
        print(f"[SA] Nghiệm ban đầu: {len(sol)} xe | "
              f"{init_dist / KM_SCALE:.2f} km")
        return sol

    # ── Core SA ───────────────────────────────────────────────────────

    def solve(self) -> tuple:
        """
        Chạy SA, trả về (best_solution, best_cost_units).
        best_cost_units là đơn vị nội bộ — dùng /100 để ra km.
        """
        current_sol  = self.initial_solution()
        route_costs  = [self.route_cost(r) for r in current_sol]
        current_cost = sum(route_costs) + len(current_sol) * self.vehicle_penalty

        best_sol  = [r[:] for r in current_sol]
        best_cost = current_cost

        T                = self.T_start
        no_improve_count = 0
        step             = 0

        while T > self.T_min:
            if no_improve_count >= self.max_no_improve:
                print(f"\n[SA] Dừng sớm tại step {step}: "
                      f"{no_improve_count} vòng không cải thiện.")
                break

            improved_this_temp = False

            for _ in range(self.iter_per_T):
                if len(current_sol) < 2:
                    break

                idx1, idx2 = random.sample(range(len(current_sol)), 2)
                r1, r2     = current_sol[idx1], current_sol[idx2]

                if len(r1) <= 2:
                    continue

                move_type     = random.random()
                accepted_move = False

                # SWAP: đổi chỗ 2 khách giữa 2 route
                if move_type < 0.4:
                    if len(r2) <= 2:
                        continue
                    i = random.randint(1, len(r1) - 2)
                    j = random.randint(1, len(r2) - 2)
                    load_r1 = (self.get_route_load(r1)
                               - self.demands_map.get(r1[i], self.demand)
                               + self.demands_map.get(r2[j], self.demand))
                    load_r2 = (self.get_route_load(r2)
                               - self.demands_map.get(r2[j], self.demand)
                               + self.demands_map.get(r1[i], self.demand))
                    if load_r1 > self.capacity or load_r2 > self.capacity:
                        continue
                    r1[i], r2[j] = r2[j], r1[i]
                    accepted_move = True
                    old_costs     = (route_costs[idx1], route_costs[idx2])

                # RELOCATE: di chuyển 1 khách sang route khác
                elif move_type < 0.8:
                    i      = random.randint(1, len(r1) - 2)
                    node   = r1[i]
                    d_node = self.demands_map.get(node, self.demand)
                    if self.get_route_load(r2) + d_node > self.capacity:
                        continue
                    j = random.randint(1, len(r2) - 1)
                    r1.pop(i)
                    r2.insert(j, node)
                    accepted_move = True
                    old_costs     = (route_costs[idx1], route_costs[idx2])

                # Thay thế block: # 2-OPT nội tuyến
                else:
                    if len(r1) <= 3: # Phải có ít nhất 2 khách hàng mới có thể đổi chỗ
                        continue
                    # INTRA-ROUTE SWAP: Đổi chỗ 2 khách hàng trong CÙNG 1 tuyến
                    i, j = random.sample(range(1, len(r1) - 1), 2)
                    r1[i], r1[j] = r1[j], r1[i]
                    
                    accepted_move = True
                    old_costs     = (route_costs[idx1],)

                if not accepted_move:
                    continue

                new_r1 = self.route_cost(r1)
                new_r2 = self.route_cost(r2) if move_type < 0.8 else None

                v_delta = 0
                if len(r1) <= 2:
                    v_delta -= self.vehicle_penalty
                if move_type < 0.8 and new_r2 is not None and len(r2) <= 2:
                    v_delta -= self.vehicle_penalty

                new_total = (
                    current_cost - old_costs[0] + new_r1
                    if move_type >= 0.8
                    else current_cost - sum(old_costs) + new_r1 + (new_r2 or 0) + v_delta
                )

                delta  = new_total - current_cost
                accept = delta < 0 or random.random() < math.exp(-min(delta / T, 700))

                if accept:
                    current_cost      = new_total
                    route_costs[idx1] = new_r1
                    if move_type < 0.8 and new_r2 is not None:
                        route_costs[idx2] = new_r2
                    if len(r1) <= 2:
                        current_sol.pop(idx1)
                        route_costs.pop(idx1)
                        if idx2 > idx1:
                            idx2 -= 1
                    if current_cost < best_cost:
                        best_sol           = [r[:] for r in current_sol]
                        best_cost          = current_cost
                        improved_this_temp = True
                else:
                    # Rollback
                    if move_type < 0.4:
                        r1[i], r2[j] = r2[j], r1[i]
                    elif move_type < 0.8:
                        node = r2.pop(j)
                        r1.insert(i, node)
                    else:
                        # Rollback cho Intra-route Swap (Đổi lại vị trí cũ)
                        r1[i], r1[j] = r1[j], r1[i]

            no_improve_count = 0 if improved_this_temp else no_improve_count + 1
            T    *= self.alpha
            step += 1

            if step % 1000 == 0:
                actual_v    = len([r for r in best_sol if len(r) > 2])
                actual_dist = sum(self.route_cost(r) for r in best_sol if len(r) > 2)
                # [FIX-4] Chia KM_SCALE=100 để ra km đúng
                print(f"Step {step:6d} | T={T:.2f} | "
                      f"Best={actual_dist / KM_SCALE:.2f} km | "
                      f"Xe={actual_v} | "
                      f"NoImprove={no_improve_count}/{self.max_no_improve}")

        # [FIX-5] Trả về đơn vị nội bộ, Pipeline.build_result() sẽ chia /100
        best_dist_units = sum(self.route_cost(r) for r in best_sol if len(r) > 2)
        return best_sol, best_dist_units