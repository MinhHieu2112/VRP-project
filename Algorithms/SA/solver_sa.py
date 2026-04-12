

import random
import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Algorithms.Init_strategies.Init_strategies import random_init, _build_demands


class SimulatedAnnealingSolver:
    def __init__(self, data_bundle: dict, config: dict):
        """Khởi tạo SA solver với ma trận khoảng cách và cấu hình."""
        self.dist = data_bundle['distance_matrix']
        self.n    = len(self.dist)

        cons           = config.get('constraints', {})
        self.capacity  = cons.get('vehicle_capacity', 10)
        self.demand    = cons.get('default_demand', 1)

        sa_cfg              = config.get('alns_parameters', {})
        self.T_start        = sa_cfg.get('start_temperature', 5000)
        self.T_min          = sa_cfg.get('end_temperature', 0.1)
        self.alpha          = sa_cfg.get('step', 0.9999)
        self.max_no_improve = sa_cfg.get('max_no_improve', 1000)

        self.iter_per_T    = 500
        self.vehicle_penalty = 1000

        # [FIX-SA-2] Dùng demands_map dict thay vì len(route)-2
        self.demands_map = _build_demands(
            self.n, demands=None, default_demand=float(self.demand)
        )

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def route_cost(self, route: list) -> float:
        """Tính tổng khoảng cách một route (mét)."""
        if len(route) < 2:
            return 0.0
        return sum(self.dist[route[i]][route[i + 1]]
                   for i in range(len(route) - 1))

    def get_route_load(self, route: list) -> float:
        """
        [FIX-SA-2] Tính tổng demand của route từ demands_map.
        Không dùng len(route)-2 vì không tổng quát khi demand != 1.
        """
        return sum(self.demands_map.get(n, self.demand)
                   for n in route if n != 0)

    def _solution_cost(self, solution: list) -> float:
        """Tổng chi phí = tổng khoảng cách + penalty số xe."""
        return (sum(self.route_cost(r) for r in solution)
                + len(solution) * self.vehicle_penalty)

    # ──────────────────────────────────────────────────────────────────
    # Khởi tạo nghiệm (dùng random_init từ init_strategies)
    # ──────────────────────────────────────────────────────────────────

    def initial_solution(self) -> list:
        """Tạo nghiệm ban đầu bằng Greedy Nearest Neighbor (NNH)."""
        return random_init(
            matrix        = self.dist,
            num_nodes     = self.n,
            capacity      = self.capacity,
            demands       = self.demands_map,
            default_demand= self.demand,
        )

    # ──────────────────────────────────────────────────────────────────
    # Main solve
    # ──────────────────────────────────────────────────────────────────

    def solve(self):
        """
        Chạy SA và trả về (best_solution, best_distance_meters).
        Dừng khi T < T_min HOẶC không cải thiện sau max_no_improve bước nhiệt độ.
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
                print(f"\n[SA] Dừng: {no_improve_count} vòng không cải thiện "
                      f"(ngưỡng={self.max_no_improve}).")
                break

            improved_this_temp = False

            for _ in range(self.iter_per_T):
                if len(current_sol) < 2:
                    break

                idx1, idx2 = random.sample(range(len(current_sol)), 2)
                r1 = current_sol[idx1]
                r2 = current_sol[idx2]

                if len(r1) <= 2:
                    continue

                move_type     = random.random()
                accepted_move = False

                # ── Move 1: SWAP ──────────────────────────────────────
                if move_type < 0.4:
                    if len(r2) <= 2:
                        continue
                    i = random.randint(1, len(r1) - 2)
                    j = random.randint(1, len(r2) - 2)

                    # Kiểm tra capacity SAU khi swap
                    load_r1_after = (self.get_route_load(r1)
                                     - self.demands_map.get(r1[i], self.demand)
                                     + self.demands_map.get(r2[j], self.demand))
                    load_r2_after = (self.get_route_load(r2)
                                     - self.demands_map.get(r2[j], self.demand)
                                     + self.demands_map.get(r1[i], self.demand))

                    if (load_r1_after > self.capacity
                            or load_r2_after > self.capacity):
                        continue

                    r1[i], r2[j] = r2[j], r1[i]
                    accepted_move = True
                    old_costs     = (route_costs[idx1], route_costs[idx2])

                # ── Move 2: RELOCATE ──────────────────────────────────
                elif move_type < 0.8:
                    i = random.randint(1, len(r1) - 2)
                    node    = r1[i]
                    d_node  = self.demands_map.get(node, self.demand)

                    # [FIX-SA-1] Kiểm tra capacity DUY NHẤT ở đây,
                    # rồi mới thực hiện pop/insert. Không kiểm tra 2 lần.
                    if self.get_route_load(r2) + d_node > self.capacity:
                        continue

                    j = random.randint(1, len(r2) - 1)
                    r1.pop(i)
                    r2.insert(j, node)
                    accepted_move = True
                    old_costs     = (route_costs[idx1], route_costs[idx2])

                # ── Move 3: 2-OPT intra-route ─────────────────────────
                else:
                    if len(r1) <= 3:
                        continue
                    i = random.randint(1, len(r1) - 2)
                    j = random.randint(i + 1, len(r1) - 1)
                    # [FIX-SA-3] list() bọc reversed() cho slice assignment
                    r1[i:j] = list(reversed(r1[i:j]))
                    accepted_move = True
                    old_costs     = (route_costs[idx1],)

                if not accepted_move:
                    continue

                # Tính chi phí mới
                new_r1_cost = self.route_cost(r1)
                new_r2_cost = self.route_cost(r2) if move_type < 0.8 else None

                v_penalty_delta = 0
                if len(r1) <= 2:
                    v_penalty_delta -= self.vehicle_penalty
                if move_type < 0.8 and new_r2_cost is not None and len(r2) <= 2:
                    v_penalty_delta -= self.vehicle_penalty

                if move_type >= 0.8:
                    new_total = current_cost - old_costs[0] + new_r1_cost
                else:
                    new_total = (current_cost - sum(old_costs)
                                 + new_r1_cost + (new_r2_cost or 0)
                                 + v_penalty_delta)

                delta  = new_total - current_cost
                accept = (delta < 0
                          or random.random() < math.exp(-min(delta / T, 700)))

                if accept:
                    current_cost      = new_total
                    route_costs[idx1] = new_r1_cost
                    if move_type < 0.8 and new_r2_cost is not None:
                        route_costs[idx2] = new_r2_cost

                    # Xóa route rỗng [0, 0]
                    if len(r1) <= 2:
                        current_sol.pop(idx1)
                        route_costs.pop(idx1)
                        # [FIX-SA-1] Điều chỉnh idx2 sau khi xóa idx1
                        if idx2 > idx1:
                            idx2 -= 1

                    if current_cost < best_cost:
                        best_sol              = [r[:] for r in current_sol]
                        best_cost             = current_cost
                        improved_this_temp    = True

                else:
                    # Rollback
                    if move_type < 0.4:
                        r1[i], r2[j] = r2[j], r1[i]
                    elif move_type < 0.8:
                        # [FIX-SA-1] Pop từ r2 rồi insert lại r1
                        node = r2.pop(j)
                        r1.insert(i, node)
                    else:
                        # [FIX-SA-3] list() bọc reversed() khi rollback
                        r1[i:j] = list(reversed(r1[i:j]))

            no_improve_count = 0 if improved_this_temp else no_improve_count + 1
            T    *= self.alpha
            step += 1

            if step % 1000 == 0:
                actual_v    = len([r for r in best_sol if len(r) > 2])
                actual_dist = sum(self.route_cost(r)
                                  for r in best_sol if len(r) > 2)
                print(f"Step {step:6d} | T={T:.4f} | "
                      f"Best={actual_dist:.0f}m ({actual_dist/1000:.2f}km) | "
                      f"Xe={actual_v} | "
                      f"NoImprove={no_improve_count}/{self.max_no_improve}")

        actual_best_dist = sum(self.route_cost(r)
                               for r in best_sol if len(r) > 2)
        return best_sol, actual_best_dist