# File định nghĩa lớp biểu diễn trạng thái lời giải CvrpState phục vụ cho ALNS.
import copy
import sys
from alns import State
import numpy as np

class CvrpState(State):
    """Trạng thái lời giải CVRP chứa danh sách các tuyến đường và khách hàng chưa gán."""

    def __init__(self, routes, unassigned, distance_matrix, capacity, demands, config):
        # Khởi tạo các thuộc tính của trạng thái lời giải và tính toán tải trọng các tuyến.
        self.routes = routes
        self.unassigned = unassigned
        self.distance_matrix = distance_matrix
        self.capacity = capacity
        self.demands = demands
        self.config = config
        self.route_loads = [sum(demands[node] for node in r if node != 0) for r in routes]

    def objective(self):
        # Tính tổng quãng đường di chuyển cộng với các khoản phạt vi phạm ràng buộc.
        total_distance = sum(self.route_cost(route) for route in self.routes)

        constraints = self.config.get("constraints", {})
        penalty_node    = constraints.get("penalty_unassigned", 50_000_000)
        penalty_v_over  = constraints.get("penalty_vehicle_over", 5_000_000)
        max_v           = constraints.get("max_vehicles", 200)

        num_unassigned = len(self.unassigned)
        total_penalty  = num_unassigned * penalty_node

        actual_vehicles = len([r for r in self.routes if len(r) > 2])
        vehicle_penalty = 0
        if actual_vehicles > max_v:
            vehicle_penalty = (actual_vehicles - max_v) * penalty_v_over

        return total_distance + total_penalty + vehicle_penalty

    def apply_2opt(self):
        # Thực hiện thuật toán Or-opt nội tuyến làm mịn lộ trình mà không làm đảo cung bất đối xứng.
        import sys

        new_routes = []
        active = [r for r in self.routes if len(r) > 2]
        total  = len(active)

        for idx, route in enumerate(active):
            percent  = (idx + 1) / total * 100
            progress = int(percent / 5)
            bar      = "█" * progress + "-" * (20 - progress)
            sys.stdout.write(
                f"\r  Progress: [{bar}] {percent:.1f}% (Xe {idx+1}/{total})")
            sys.stdout.flush()

            if len(route) <= 3:
                new_routes.append(route)
                continue

            best_route = route[:]
            improved   = True
            max_iters  = 50

            while improved and max_iters > 0:
                improved  = False
                max_iters -= 1

                for i in range(1, len(best_route) - 1):
                    node  = best_route[i]
                    prev_i = best_route[i - 1]
                    next_i = best_route[i + 1]

                    cost_remove = (
                        self.distance_matrix[prev_i, node]
                        + self.distance_matrix[node, next_i]
                        - self.distance_matrix[prev_i, next_i]
                    )

                    best_gain = 0
                    best_j    = -1

                    for j in range(1, len(best_route) - 1):
                        if j == i or j == i - 1:
                            continue
                        prev_j = best_route[j - 1]
                        next_j = best_route[j]

                        cost_insert = (
                            self.distance_matrix[prev_j, node]
                            + self.distance_matrix[node, next_j]
                            - self.distance_matrix[prev_j, next_j]
                        )

                        gain = cost_remove - cost_insert
                        if gain > best_gain + 1e-6:
                            best_gain = gain
                            best_j    = j

                    if best_j != -1:
                        route_tmp = best_route[:]
                        route_tmp.pop(i)
                        insert_at = best_j if best_j < i else best_j - 1
                        route_tmp.insert(insert_at, node)
                        best_route = route_tmp
                        improved   = True
                        break

            new_routes.append(best_route)

        self.routes = new_routes
        self.route_loads = [
            sum(self.demands[node] for node in r if node != 0)
            for r in self.routes
        ]
        print("\n[XONG] Đã xong toàn bộ lộ trình.")

    def copy(self):
        # Sao chép sâu trạng thái lời giải hiện tại.
        new_state = object.__new__(CvrpState)
        new_state.routes            = [r[:] for r in self.routes]
        new_state.unassigned        = self.unassigned[:]
        new_state.distance_matrix   = self.distance_matrix
        new_state.capacity          = self.capacity
        new_state.demands           = self.demands
        new_state.config            = self.config
        new_state.route_loads       = self.route_loads[:]
        return new_state

    @property
    def cost(self):
        # Lấy giá trị hàm mục tiêu của trạng thái.
        return self.objective()

    def get_route_load(self, route_idx):
        # Lấy tổng tải trọng của tuyến đường được chỉ định.
        return self.route_loads[route_idx]

    def is_valid(self):
        # Kiểm tra xem phương án hiện tại có vi phạm ràng buộc tải trọng hay không.
        for i, route in enumerate(self.routes):
            if self.route_loads[i] > self.capacity:
                return False
        return True

    def route_cost(self, route):
        # Tính toán chi phí khoảng cách thực tế của một tuyến đường cụ thể.
        cost = 0
        for i in range(len(route) - 1):
            cost += self.distance_matrix[route[i], route[i + 1]]
        return cost
