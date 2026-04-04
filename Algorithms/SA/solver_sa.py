import random
import math
import time

class SimulatedAnnealingSolver:
    def __init__(self, data_bundle, config):
        self.dist = data_bundle['distance_matrix']
        self.n = len(self.dist)

        cons = config.get('constraints', {})
        self.capacity = cons.get('vehicle_capacity', 10)
        self.demand = cons.get('default_demand', 1)

        sa_cfg = config.get('alns_parameters', {})
        self.max_runtime = sa_cfg.get('max_runtime', 180)
        self.T_start = sa_cfg.get('start_temperature', 5000)
        self.T_min = sa_cfg.get('end_temperature', 0.1)
        self.alpha = sa_cfg.get('step', 0.9995)

        # Tăng số vòng lặp mỗi bước nhiệt độ vì đã tối ưu tốc độ
        self.iter_per_T = 500 
        self.vehicle_penalty = 1000 # Tăng penalty để ép giảm số xe hiệu quả hơn

    def route_cost(self, route):
        if len(route) < 2: return 0
        cost = 0
        for i in range(len(route) - 1):
            cost += self.dist[route[i]][route[i+1]]
        return cost

    def get_route_load(self, route):
        return (len(route) - 2) * self.demand

    def initial_solution(self):
        """Khởi tạo bằng Greedy Nearest Neighbor"""
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
        start_time = time.time()
        current_sol = self.initial_solution()
        # Tính chi phí ban đầu từng route để tối ưu Delta
        route_costs = [self.route_cost(r) for r in current_sol]
        current_cost = sum(route_costs) + len(current_sol) * self.vehicle_penalty
        
        best_sol = [r[:] for r in current_sol]
        best_cost = current_cost

        T = self.T_start
        step = 0

        while T > self.T_min:
            if time.time() - start_time > self.max_runtime:
                break

            for _ in range(self.iter_per_T):
                # Chọn ngẫu nhiên 2 route (có thể trùng nhau)
                idx1, idx2 = random.sample(range(len(current_sol)), 2)
                r1, r2 = current_sol[idx1], current_sol[idx2]
                
                # Tránh các route rỗng
                if len(r1) <= 2: continue

                move_type = random.random()
                delta = 0
                
                
                if move_type >= 0.8:  # 2-opt
                    old_costs = (route_costs[idx1],)
                else:
                    old_costs = (route_costs[idx1], route_costs[idx2])
                
                # --- THỰC HIỆN MOVE ---
                accepted_move = False
                
                # 1. SWAP (Hoán đổi 2 khách hàng)
                if move_type < 0.4:
                    i = random.randint(1, len(r1) - 2)
                    j = random.randint(1, len(r2) - 2)
                    # Swap thực tế
                    r1[i], r2[j] = r2[j], r1[i]
                    accepted_move = True # Swap cùng demand nên load ko đổi
                
                # 2. RELOCATE (Chuyển 1 khách hàng sang route khác)
                elif move_type < 0.8:
                    i = random.randint(1, len(r1) - 2)
                    j = random.randint(1, len(r2) - 1)
                    if self.get_route_load(r2) + self.demand <= self.capacity:
                        node = r1.pop(i)
                        r2.insert(j, node)
                        accepted_move = True
                
                # 3. 2-OPT (Đảo ngược đoạn đường - chỉ nội bộ route)
                else:
                    if len(r1) > 3:
                        i = random.randint(1, len(r1) - 2)
                        j = random.randint(i + 1, len(r1) - 1)
                        r1[i:j] = reversed(r1[i:j])
                        accepted_move = True

                if accepted_move:
                    # Tính chi phí mới chỉ cho 2 route bị đổi
                    new_r1_cost = self.route_cost(r1)
                    if move_type < 0.8:  # chỉ khi có r2 thay đổi
                        new_r2_cost = self.route_cost(r2)
                    
                    # Tính Delta bao gồm cả penalty nếu có route bị xóa hoàn toàn
                    v_penalty_delta = 0
                    if len(r1) <= 2: v_penalty_delta -= self.vehicle_penalty
                    if len(r2) <= 2: v_penalty_delta -= self.vehicle_penalty # Trường hợp hiếm
                    
                    if move_type >= 0.8:  # 2-opt
                        new_total_cost = current_cost - old_costs[0] + new_r1_cost
                    else:
                        new_total_cost = (current_cost - sum(old_costs)) + (new_r1_cost + new_r2_cost) + v_penalty_delta
                    delta = new_total_cost - current_cost

                    # Chấp nhận theo quy tắc SA
                    if delta < 0 or random.random() < math.exp(-min(delta / T, 700)):
                        current_cost = new_total_cost
                        route_costs[idx1] = new_r1_cost
                        if move_type < 0.8:  # chỉ update r2 nếu có thay đổi
                            route_costs[idx2] = new_r2_cost
                        
                        # Xóa route rỗng nếu có
                        if len(r1) <= 2:
                            current_sol.pop(idx1)
                            route_costs.pop(idx1)
                            if idx2 > idx1:
                                idx2 -= 1
                        
                        if current_cost < best_cost:
                            best_sol = [r[:] for r in current_sol]
                            best_cost = current_cost
                    else:
                        # ROLLBACK (Quay lại trạng thái cũ nếu ko chấp nhận)
                        if move_type < 0.4: # Rollback Swap
                            r1[i], r2[j] = r2[j], r1[i]
                        elif move_type < 0.8: # Rollback Relocate
                            node = r2.pop(j)
                            r1.insert(i, node)
                        else: # Rollback 2-opt
                            r1[i:j] = reversed(r1[i:j])

            T *= self.alpha
            step += 1
            if step % 1000 == 0:
                # Tính số xe thực tế
                actual_v = len([r for r in best_sol if len(r) > 2])
                # Tính quãng đường thực tế (không penalty)
                current_actual_dist = sum(self.route_cost(r) for r in best_sol if len(r) > 2)
                print(f"Step {step}, T={T:.4f}, Best Dist={current_actual_dist:.2f}, Vehicles={actual_v}")

        # Tính toán lại chi phí thực tế (không kèm penalty xe) để trả về cho main.py
        actual_best_dist = sum(self.route_cost(r) for r in best_sol if len(r) > 2)
        
        # Trả về 2 giá trị để khớp với main.py: routes, total_cost
        return best_sol, actual_best_dist