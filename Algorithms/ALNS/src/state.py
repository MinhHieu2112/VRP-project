# Định nghĩa lớp CvrpState lưu trữ trạng thái của lời giải CVRP trong thuật toán ALNS.
import copy
import sys
from alns import State
import numpy as np
from Utils.Operators.local_search import route_cost as _route_cost, or_opt_intra

class CvrpState(State):
    # Lớp đại diện cho trạng thái lời giải CVRP chứa danh sách các tuyến đường và chi phí tương ứng.

    def __init__(self, routes, unassigned, distance_matrix, capacity, demands, config):
        # Khởi tạo các thuộc tính của trạng thái lời giải và tính toán tải trọng cùng chi phí các tuyến.
        self.routes = routes
        self.unassigned = unassigned
        self.distance_matrix = distance_matrix
        self.capacity = capacity
        self.demands = demands
        self.config = config
        self.route_loads = [sum(demands[node] for node in r if node != 0) for r in routes]
        self.route_costs = [self.route_cost(r) for r in routes]

    def objective(self):
        # Lấy tổng chi phí di chuyển đã được cache cộng với các khoản phạt vi phạm ràng buộc.
        total_distance = sum(self.route_costs)

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
        # Tối ưu hóa nội tuyến tất cả các tuyến đường bằng cách gọi Utils.local_search.or_opt_intra.
        import sys
        active = [r for r in self.routes if len(r) > 2]
        total  = len(active)
        new_routes = []

        for idx, route in enumerate(active):
            percent  = (idx + 1) / total * 100
            progress = int(percent / 5)
            bar      = "█" * progress + "-" * (20 - progress)
            sys.stdout.write(
                f"\r  Progress: [{bar}] {percent:.1f}% (Xe {idx+1}/{total})")
            sys.stdout.flush()
            new_routes.append(or_opt_intra(self.distance_matrix, route))

        self.routes = new_routes
        self.route_loads = [
            sum(self.demands[node] for node in r if node != 0)
            for r in self.routes
        ]
        self.route_costs = [self.route_cost(r) for r in self.routes]
        print("\n[XONG] Đã xong toàn bộ lộ trình.")

    def copy(self):
        # Sao chép sâu trạng thái lời giải hiện tại cùng cache khoảng cách và tải trọng.
        new_state = object.__new__(CvrpState)
        new_state.routes            = [r[:] for r in self.routes]
        new_state.unassigned        = self.unassigned[:]
        new_state.distance_matrix   = self.distance_matrix
        new_state.capacity          = self.capacity
        new_state.demands           = self.demands
        new_state.config            = self.config
        new_state.route_loads       = self.route_loads[:]
        new_state.route_costs       = self.route_costs[:]
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
        # Gọi Utils.local_search.route_cost để tính chi phí một tuyến đường (tương thích bất đối xứng).
        return _route_cost(self.distance_matrix, route)
