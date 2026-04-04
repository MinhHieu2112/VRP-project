import copy
import sys
from alns import State
import numpy as np
 
class CvrpState(State):
    def __init__(self, routes, unassigned, distance_matrix, capacity, demands, config):
        self.routes = routes
        self.unassigned = unassigned
        self.distance_matrix = distance_matrix
        self.capacity = capacity
        self.demands = demands
        self.config = config
        self.route_loads = [sum(demands[node] for node in r) for r in routes]
 
    def objective(self):
        total_distance = sum(self.route_cost(route) for route in self.routes)
 
        constraints = self.config.get("constraints", {})
        penalty_node = constraints.get("penalty_unassigned", 1000000)
        penalty_v_over = constraints.get("penalty_vehicle_over", 500000)
        max_v = constraints.get("max_vehicles", 200)
 
        num_unassigned = len(self.unassigned)
        total_penalty = num_unassigned * penalty_node
 
        actual_vehicles = len([r for r in self.routes if len(r) > 2])
        vehicle_penalty = 0
        if actual_vehicles > max_v:
            vehicle_penalty = (actual_vehicles - max_v) * penalty_v_over
 
        return total_distance + total_penalty + vehicle_penalty
 
    def apply_2opt(self):
        """
        2-opt với Progress Bar thủ công để không bị cảm giác đơ máy.
        """
        new_routes = []
        actual_vehicles = [r for r in self.routes if len(r) > 2]
        total = len(actual_vehicles)
        
        print(f"Đang tối ưu {total} xe:")
        
        for idx, route in enumerate(actual_vehicles):
            # In thanh tiến độ đơn giản
            percent = (idx + 1) / total * 100
            progress = int(percent / 5)
            bar = "█" * progress + "-" * (20 - progress)
            sys.stdout.write(f"\r  Progress: [{bar}] {percent:.1f}% (Xe {idx+1}/{total})")
            sys.stdout.flush()

            if len(route) <= 3:
                new_routes.append(route)
                continue

            best_route = route[:]
            improved = True
            max_inner_iters = 100
            count = 0

            while improved and count < max_inner_iters:
                improved = False
                for i in range(1, len(best_route) - 2):
                    for j in range(i + 1, len(best_route) - 1):
                        A, B = best_route[i-1], best_route[i]
                        C, D = best_route[j], best_route[j+1]

                        old_edge = self.distance_matrix[A, B] + self.distance_matrix[C, D]
                        new_edge = self.distance_matrix[A, C] + self.distance_matrix[B, D]

                        if new_edge < old_edge - 0.00001:
                            best_route[i:j+1] = reversed(best_route[i:j+1])
                            improved = True
                count += 1
            
            new_routes.append(best_route)
        
        self.routes = new_routes
        print("\n[XONG] Đã tối ưu xong toàn bộ lộ trình.")
 
    def copy(self):
        new_state = CvrpState(
            [r[:] for r in self.routes],
            self.unassigned[:],
            self.distance_matrix,
            self.capacity,
            self.demands,
            self.config
        )
        # Copy cả bộ nhớ đệm tải trọng
        new_state.route_loads = self.route_loads[:]
        return new_state
 
    @property
    def cost(self):
        return self.objective()
 
    def get_route_load(self, route_idx):
        return self.route_loads[route_idx]
 
    def is_valid(self):
        for route in self.routes:
            if self.get_route_load(route) > self.capacity:
                return False
        return True
 
    def route_cost(self, route):
        cost = 0
        for i in range(len(route) - 1):
            cost += self.distance_matrix[route[i], route[i+1]]
        return cost