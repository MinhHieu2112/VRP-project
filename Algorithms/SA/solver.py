# Lớp điều phối chính của thuật toán Simulated Annealing thực thi vòng lặp tối ưu hóa.
from __future__ import annotations
import math
import random
import numpy as np

from typing import Dict, List, Tuple
from Algorithms.Init_strategies.Init_strategies import init_solution
from Algorithms.SA.config_handler import load_sa_config, SAConfig
from Algorithms.SA.cost_evaluator import route_cost, build_route_costs

KM_SCALE = 100


class SimulatedAnnealingSolver:
    """Lớp thực hiện thuật toán Simulated Annealing giải bài toán VRP."""

    def __init__(self, data: dict, config: dict | SAConfig):
        # Khởi tạo các tham số, ràng buộc và dữ liệu ma trận khoảng cách cho SA.
        self.dist = data["distance_matrix"]
        self.n = self.dist.shape[0]

        if isinstance(config, SAConfig):
            self.cfg = config
        else:
            self.cfg = load_sa_config(config)

        self.capacity = self.cfg.capacity
        self.demand = self.cfg.demand
        self.max_v = self.cfg.max_v
        self.T_start = self.cfg.T_start
        self.T_min = self.cfg.T_min
        self.alpha = self.cfg.alpha
        self.max_no_improve = self.cfg.max_no_improve
        self.iter_per_T = self.cfg.iter_per_T
        self.init_strategy = self.cfg.init_strategy
        self.vehicle_penalty = self.cfg.vehicle_penalty

        self.demands_map = {
            i: (0.0 if i == 0 else float(self.demand))
            for i in range(self.n)
        }

    def initial_solution(self) -> List[List[int]]:
        # Khởi tạo phương án sơ bộ ban đầu dựa trên chiến lược được chọn.
        print(f"[SA] Khởi tạo nghiệm bằng: '{self.init_strategy}'")

        sol = init_solution(
            strategy=self.init_strategy,
            matrix=self.dist,
            num_nodes=self.n,
            capacity=self.capacity,
            demands=self.demands_map,
            default_demand=float(self.demand),
            max_vehicles=self.max_v,
            validate=True,
        )

        init_dist = sum(route_cost(self.dist, r) for r in sol if len(r) > 2)
        print(f"[SA] Nghiệm ban đầu: {len(sol)} xe | {init_dist / KM_SCALE:.2f} km")
        return sol

    def solve(self) -> Tuple[List[List[int]], float]:
        # Thực hiện vòng lặp mô phỏng luyện kim tối ưu với Delta Evaluation O(1) để tìm lời giải tốt nhất.
        current_sol = self.initial_solution()
        route_costs = build_route_costs(self.dist, current_sol)
        route_loads = [sum(self.demands_map[node] for node in r) for r in current_sol]
        current_cost = sum(route_costs) + len(current_sol) * self.vehicle_penalty

        best_sol = [r[:] for r in current_sol]
        best_cost = current_cost

        T = self.T_start
        no_improve_count = 0
        step = 0

        # Import trực tiếp các hàm đánh giá delta O(1)
        from Algorithms.SA.cost_evaluator import (
            eval_swap_delta, eval_relocate_delta, eval_intra_swap_delta
        )

        while T > self.T_min:
            if no_improve_count >= self.max_no_improve:
                print(
                    f"\n[SA] Dừng sớm tại step {step}: {no_improve_count} vòng không cải thiện."
                )
                break

            improved_this_temp = False

            for _ in range(self.iter_per_T):
                if len(current_sol) < 2:
                    break

                idx1, idx2 = random.sample(range(len(current_sol)), 2)
                r1, r2 = current_sol[idx1], current_sol[idx2]

                if len(r1) <= 2:
                    continue

                move_type = random.random()

                if move_type < 0.4:
                    # Phép thử Inter-route Swap (tráo đổi 2 node giữa 2 tuyến)
                    if len(r2) <= 2:
                        continue
                    i = random.randint(1, len(r1) - 2)
                    j = random.randint(1, len(r2) - 2)
                    u, v = r1[i], r2[j]

                    new_load1 = route_loads[idx1] - self.demands_map[u] + self.demands_map[v]
                    new_load2 = route_loads[idx2] - self.demands_map[v] + self.demands_map[u]

                    if new_load1 > self.capacity or new_load2 > self.capacity:
                        continue

                    delta_r1, delta_r2 = eval_swap_delta(self.dist, r1, r2, i, j)
                    delta_cost = delta_r1 + delta_r2

                    accept = delta_cost < 0 or random.random() < math.exp(-min(delta_cost / T, 700.0))
                    if accept:
                        r1[i], r2[j] = r2[j], r1[i]
                        route_loads[idx1] = new_load1
                        route_loads[idx2] = new_load2
                        route_costs[idx1] += delta_r1
                        route_costs[idx2] += delta_r2
                        current_cost += delta_cost

                        if current_cost < best_cost:
                            best_sol = [r[:] for r in current_sol]
                            best_cost = current_cost
                            improved_this_temp = True

                elif move_type < 0.8:
                    # Phép thử Inter-route Relocate (chuyển 1 node sang tuyến xe khác)
                    i = random.randint(1, len(r1) - 2)
                    u = r1[i]

                    new_load2 = route_loads[idx2] + self.demands_map[u]
                    if new_load2 > self.capacity:
                        continue

                    ins_pos = random.randint(1, len(r2) - 1)
                    delta_r1, delta_r2 = eval_relocate_delta(self.dist, r1, r2, i, ins_pos)
                    delta_cost = delta_r1 + delta_r2

                    # Phạt số lượng xe nếu r1 trở thành rỗng (chỉ còn [0, 0])
                    v_delta = 0
                    if len(r1) == 3:
                        v_delta = -self.vehicle_penalty
                    delta_cost += v_delta

                    accept = delta_cost < 0 or random.random() < math.exp(-min(delta_cost / T, 700.0))
                    if accept:
                        r1.pop(i)
                        r2.insert(ins_pos, u)
                        route_loads[idx1] -= self.demands_map[u]
                        route_loads[idx2] = new_load2
                        route_costs[idx1] += delta_r1
                        route_costs[idx2] += delta_r2
                        current_cost += delta_cost

                        if len(r1) <= 2:
                            current_sol.pop(idx1)
                            route_loads.pop(idx1)
                            route_costs.pop(idx1)

                        if current_cost < best_cost:
                            best_sol = [r[:] for r in current_sol]
                            best_cost = current_cost
                            improved_this_temp = True

                else:
                    # Phép thử Intra-route Swap (tráo đổi 2 node nội bộ tuyến xe)
                    if len(r1) < 4:
                        continue
                    i, j = random.sample(range(1, len(r1) - 1), 2)
                    delta_cost = eval_intra_swap_delta(self.dist, r1, i, j)

                    accept = delta_cost < 0 or random.random() < math.exp(-min(delta_cost / T, 700.0))
                    if accept:
                        r1[i], r1[j] = r1[j], r1[i]
                        route_costs[idx1] += delta_cost
                        current_cost += delta_cost

                        if current_cost < best_cost:
                            best_sol = [r[:] for r in current_sol]
                            best_cost = current_cost
                            improved_this_temp = True

            no_improve_count = 0 if improved_this_temp else no_improve_count + 1
            T *= self.alpha
            step += 1

            if step % 1000 == 0:
                actual_v = len([r for r in best_sol if len(r) > 2])
                actual_dist = sum(route_cost(self.dist, r) for r in best_sol if len(r) > 2)
                print(
                    f"Step {step:6d} | T={T:.2f} | "
                    f"Best={actual_dist / KM_SCALE:.2f} km | "
                    f"Xe={actual_v} | "
                    f"NoImprove={no_improve_count}/{self.max_no_improve}"
                )

        best_dist_units = sum(route_cost(self.dist, r) for r in best_sol if len(r) > 2)
        return best_sol, best_dist_units
