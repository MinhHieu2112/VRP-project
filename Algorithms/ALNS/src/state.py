import copy
import sys
from alns import State
import numpy as np


class CvrpState(State):
    def __init__(self, routes, unassigned, distance_matrix, capacity, demands, config):
        # distance_matrix đơn vị mét (int). Chia 1000 CHỈ khi xuất báo cáo.
        self.routes = routes
        self.unassigned = unassigned
        self.distance_matrix = distance_matrix
        self.capacity = capacity
        self.demands = demands
        self.config = config
        self.route_loads = [sum(demands[node] for node in r if node != 0) for r in routes]

    def objective(self):
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
        """
        Or-opt nội tuyến thay thế 2-opt cổ điển, phù hợp với ACVRP bất đối xứng.

        VẤN ĐỀ CỦA 2-OPT TRUYỀN THỐNG VỚI ACVRP:
        2-opt đảo ngược đoạn route[i:j+1], điều này chỉ đúng với ma trận đối xứng
        (d(A,B) == d(B,A)). Với ACVRP (d(i,j) ≠ d(j,i)), đảo ngược chiều đi
        thường làm chi phí tăng — đó là lý do kết quả nhảy từ ~776 lên 799 km.

        GIẢI PHÁP — Or-opt (relocate 1 node):
        Thay vì đảo đoạn, thử dịch chuyển từng node sang vị trí khác trong cùng route.
        Or-opt tương thích với ACVRP vì không đảo chiều cung.

        Đây là local search làm mịn sau ALNS — không cần chạy quá nhiều vòng.
        """
        import sys

        new_routes = []
        active = [r for r in self.routes if len(r) > 2]
        total  = len(active)

        print(f"Đang Or-opt (relocate) {total} xe:")

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
            max_iters  = 50   # giới hạn vòng lặp để không chạy quá lâu

            while improved and max_iters > 0:
                improved  = False
                max_iters -= 1

                for i in range(1, len(best_route) - 1):
                    node  = best_route[i]
                    prev_i = best_route[i - 1]
                    next_i = best_route[i + 1]

                    # Chi phí khi bỏ node ra khỏi vị trí hiện tại
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

                        # Chi phí khi chèn node vào sau vị trí j-1
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
                        # Thực hiện move: xóa node khỏi i, chèn vào best_j
                        route_tmp = best_route[:]
                        route_tmp.pop(i)
                        # Điều chỉnh index sau khi pop
                        insert_at = best_j if best_j < i else best_j - 1
                        route_tmp.insert(insert_at, node)
                        best_route = route_tmp
                        improved   = True
                        break  # restart từ đầu route sau mỗi move

            new_routes.append(best_route)

        self.routes = new_routes

        # Đồng bộ route_loads (Or-opt không thay đổi thành phần khách hàng)
        self.route_loads = [
            sum(self.demands[node] for node in r if node != 0)
            for r in self.routes
        ]

        print("\n[XONG] Đã xong toàn bộ lộ trình.")

    def copy(self):
        new_state = object.__new__(CvrpState)
        new_state.routes            = [r[:] for r in self.routes]
        new_state.unassigned        = self.unassigned[:]
        new_state.distance_matrix   = self.distance_matrix   # shared, read-only
        new_state.capacity          = self.capacity
        new_state.demands           = self.demands            # shared, read-only
        new_state.config            = self.config             # shared, read-only
        new_state.route_loads       = self.route_loads[:]    # copy cache
        return new_state

    @property
    def cost(self):
        return self.objective()

    def get_route_load(self, route_idx):
        return self.route_loads[route_idx]

    def is_valid(self):
        for i, route in enumerate(self.routes):
            if self.route_loads[i] > self.capacity:
                return False
        return True

    def route_cost(self, route):
        cost = 0
        for i in range(len(route) - 1):
            cost += self.distance_matrix[route[i], route[i + 1]]
        return cost
