import copy
from alns import State
import numpy as np

class CvrpState(State):
    """
    Đại diện cho trạng thái lời giải của bài toán CVRP.
    Cấu trúc này lưu trữ các lộ trình hiện tại và danh sách các khách hàng chưa được phục vụ.
    """

    def __init__(self, routes, unassigned, distance_matrix, capacity, demands, config):
        self.routes = routes
        self.unassigned = unassigned
        self.distance_matrix = distance_matrix
        self.capacity = capacity
        self.demands = demands
        self.config = config  # Lưu cấu hình vào state

    def objective(self):
        """
        Tính tổng chi phí dựa trên các tham số từ file cấu hình JSON.
        """
        # 1. Tính tổng quãng đường di chuyển thực tế
        total_distance = sum(self.route_cost(route) for route in self.routes)
        
        # 2. Lấy các tham số hình phạt từ file config
        # Sử dụng .get() để tránh lỗi nếu trong file JSON thiếu trường dữ liệu
        constraints = self.config.get('constraints', {})
        
        penalty_node = constraints.get('penalty_unassigned', 1000000)
        penalty_v_over = constraints.get('penalty_vehicle_over', 500000)
        max_v = constraints.get('max_vehicles', 200)

        # 3. Tính phạt cho các nốt chưa gán
        num_unassigned = len(self.unassigned)
        total_penalty = num_unassigned * penalty_node
        
        # 4. Tính phạt nếu vượt quá số xe cho phép
        actual_vehicles = len([r for r in self.routes if len(r) > 2])
        vehicle_penalty = 0
        if actual_vehicles > max_v:
            vehicle_penalty = (actual_vehicles - max_v) * penalty_v_over
            
        return total_distance + total_penalty + vehicle_penalty
    
    def apply_2opt(self):
        """Tối ưu hóa nội bộ từng route và in tiến trình."""
        new_routes = []
        total_routes = len(self.routes)
        print(f"Bắt đầu Local Search cho {total_routes} lộ trình...")

        for idx, route in enumerate(self.routes):
            if len(route) <= 3:
                new_routes.append(route)
                continue
            
            # In tiến trình sau mỗi 20 xe để theo dõi
            if idx % 20 == 0:
                print(f"  > Đang xử lý: {idx}/{total_routes} xe...")

            best_route = route[:]
            improved = True
            count = 0 
            
            while improved and count < 100: # Giới hạn 100 lần cải thiện mỗi xe để tránh treo
                improved = False
                for i in range(1, len(best_route) - 2):
                    for j in range(i + 1, len(best_route) - 1):
                        old_dist = (self.distance_matrix[best_route[i-1], best_route[i]] + 
                                    self.distance_matrix[best_route[j], best_route[j+1]])
                        new_dist = (self.distance_matrix[best_route[i-1], best_route[j]] + 
                                    self.distance_matrix[best_route[i], best_route[j+1]])
                        
                        if new_dist < old_dist:
                            best_route[i:j+1] = reversed(best_route[i:j+1])
                            improved = True
                            count += 1
                            break # Tìm thấy cải thiện thì bắt đầu lại vòng lặp cho xe này
                    if improved: break
            new_routes.append(best_route)
        
        self.routes = new_routes
        print("--- Hoàn tất Local Search ---")

    def copy(self):
        # Đảm bảo khi copy state, đối tượng config cũng được truyền sang
        return CvrpState(
            [r[:] for r in self.routes],
            self.unassigned[:],
            self.distance_matrix,
            self.capacity,
            self.demands,
            self.config
        )

    @property
    def cost(self):
        """Trả về chi phí (quãng đường) để ALNS sử dụng làm chỉ số so sánh."""
        return self.objective()

    def get_route_load(self, route):
        """Hàm bổ trợ tính tổng hàng hiện có trên một lộ trình."""
        return sum(self.demands[node] for node in route)

    def is_valid(self):
        """Kiểm tra xem lời giải hiện tại có vi phạm ràng buộc sức chứa không."""
        for route in self.routes:
            if self.get_route_load(route) > self.capacity:
                return False
        return True
    
    def route_cost(self, route):
        """Tính tổng quãng đường của một lộ trình cụ thể."""
        cost = 0
        for i in range(len(route) - 1):
            cost += self.distance_matrix[route[i], route[i+1]]
        return cost
    
    