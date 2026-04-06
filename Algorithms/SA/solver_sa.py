import random
import math


class SimulatedAnnealingSolver:
    def __init__(self, data_bundle, config):
        self.dist = data_bundle['distance_matrix']
        self.n = len(self.dist)

        cons = config.get('constraints', {})
        self.capacity = cons.get('vehicle_capacity', 10)
        self.demand = cons.get('default_demand', 1)

        sa_cfg = config.get('alns_parameters', {})
        self.T_start = sa_cfg.get('start_temperature', 5000)
        self.T_min   = sa_cfg.get('end_temperature', 0.1)
        self.alpha   = sa_cfg.get('step', 0.9995)

        # No-improvement stopping criterion (thay thế max_runtime)
        self.max_no_improve = sa_cfg.get('max_no_improve', 1000)

        self.iter_per_T    = 500
        self.vehicle_penalty = 1000

    def route_cost(self, route):
        if len(route) < 2:
            return 0
        cost = 0
        for i in range(len(route) - 1):
            cost += self.dist[route[i]][route[i + 1]]
        return cost

    def get_route_load(self, route):
        return (len(route) - 2) * self.demand

    def initial_solution(self):
        """Khởi tạo bằng Greedy Nearest Neighbor."""
        unvisited = set(range(1, self.n))
        solution = []
        while unvisited:
            route = [0]
            current_load = 0
            curr = 0
            while unvisited:
                nearest = min(unvisited, key=lambda x: self.dist[curr][x])
                if current_load + self.demand <= self.capacity:
                    route.append(nearest)
                    unvisited.remove(nearest)
                    current_load += self.demand
                    curr = nearest
                else:
                    break
            route.append(0)
            solution.append(route)
        return solution

    def solve(self):
        current_sol = self.initial_solution()
        route_costs  = [self.route_cost(r) for r in current_sol]
        current_cost = sum(route_costs) + len(current_sol) * self.vehicle_penalty

        best_sol  = [r[:] for r in current_sol]
        best_cost = current_cost

        T = self.T_start

        # --- Bộ đếm no-improvement ---
        no_improve_count = 0
        step = 0

        while T > self.T_min:
            # Kiểm tra điều kiện dừng no-improvement
            if no_improve_count >= self.max_no_improve:
                print(f"\n[SA] Dừng: {no_improve_count} vòng không cải thiện "
                      f"(ngưỡng={self.max_no_improve}).")
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

                if move_type >= 0.8:
                    old_costs = (route_costs[idx1],)
                else:
                    old_costs = (route_costs[idx1], route_costs[idx2])

                accepted_move = False

                # 1. SWAP
                if move_type < 0.4:
                    if len(r2) <= 2:
                        continue
                    i = random.randint(1, len(r1) - 2)
                    j = random.randint(1, len(r2) - 2)
                    r1[i], r2[j] = r2[j], r1[i]
                    accepted_move = True

                # 2. RELOCATE
                elif move_type < 0.8:
                    if len(r2) <= 2 and self.get_route_load(r2) + self.demand > self.capacity:
                        continue
                    i = random.randint(1, len(r1) - 2)
                    j = random.randint(1, len(r2) - 1)
                    if self.get_route_load(r2) + self.demand <= self.capacity:
                        node = r1.pop(i)
                        r2.insert(j, node)
                        accepted_move = True

                # 3. 2-OPT intra-route
                else:
                    if len(r1) > 3:
                        i = random.randint(1, len(r1) - 2)
                        j = random.randint(i + 1, len(r1) - 1)
                        r1[i:j] = reversed(r1[i:j])
                        accepted_move = True

                if accepted_move:
                    new_r1_cost = self.route_cost(r1)
                    if move_type < 0.8:
                        new_r2_cost = self.route_cost(r2)

                    v_penalty_delta = 0
                    if len(r1) <= 2:
                        v_penalty_delta -= self.vehicle_penalty
                    if move_type < 0.8 and len(r2) <= 2:
                        v_penalty_delta -= self.vehicle_penalty

                    if move_type >= 0.8:
                        new_total_cost = current_cost - old_costs[0] + new_r1_cost
                    else:
                        new_total_cost = (current_cost - sum(old_costs)
                                         + new_r1_cost + new_r2_cost
                                         + v_penalty_delta)

                    delta = new_total_cost - current_cost

                    if delta < 0 or random.random() < math.exp(-min(delta / T, 700)):
                        current_cost = new_total_cost
                        route_costs[idx1] = new_r1_cost
                        if move_type < 0.8:
                            route_costs[idx2] = new_r2_cost

                        if len(r1) <= 2:
                            current_sol.pop(idx1)
                            route_costs.pop(idx1)
                            if idx2 > idx1:
                                idx2 -= 1

                        if current_cost < best_cost:
                            best_sol  = [r[:] for r in current_sol]
                            best_cost = current_cost
                            improved_this_temp = True
                    else:
                        # Rollback
                        if move_type < 0.4:
                            r1[i], r2[j] = r2[j], r1[i]
                        elif move_type < 0.8:
                            node = r2.pop(j)
                            r1.insert(i, node)
                        else:
                            r1[i:j] = reversed(r1[i:j])

            # Cập nhật bộ đếm no-improvement theo nhiệt độ (mỗi bước T = 1 "vòng")
            if improved_this_temp:
                no_improve_count = 0
            else:
                no_improve_count += 1

            T *= self.alpha
            step += 1

            if step % 1000 == 0:
                actual_v    = len([r for r in best_sol if len(r) > 2])
                actual_dist = sum(self.route_cost(r) for r in best_sol if len(r) > 2)
                print(f"Step {step:6d} | T={T:.4f} | "
                      f"Best={actual_dist:.0f}m | Xe={actual_v} | "
                      f"NoImprove={no_improve_count}/{self.max_no_improve}")

        actual_best_dist = sum(self.route_cost(r) for r in best_sol if len(r) > 2)
        return best_sol, actual_best_dist