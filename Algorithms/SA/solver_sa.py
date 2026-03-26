import random
import math
import time

class SimulatedAnnealingSolver:
    def __init__(self, data_bundle, config):
        """
        Khởi tạo Solver Simulated Annealing cho bài toán VRP.
        """
        self.dist = data_bundle['distance_matrix']
        self.n = len(self.dist)
        
        # Đọc các ràng buộc từ config
        cons = config.get('constraints', {})
        self.capacity = cons.get('vehicle_capacity', 10)
        self.demand = cons.get('default_demand', 1)
        
        # Đọc tham số thuật toán SA
        sa_cfg = config.get('alns_parameters', {})
        self.max_runtime = sa_cfg.get('max_runtime', 180)
        self.T_start = sa_cfg.get('start_temperature', 10000)
        self.T_min = sa_cfg.get('end_temperature', 0.1)
        self.alpha = sa_cfg.get('step', 0.9997)
        
        # Số vòng lặp tại mỗi mức nhiệt độ (Tăng lên để tìm kiếm kỹ hơn cho 1600 điểm)
        self.iter_per_T = 250 

    def route_cost(self, route):
        """Tính tổng quãng đường của một lộ trình cụ thể."""
        if len(route) < 2: return 0
        return sum(self.dist[route[i]][route[i + 1]] for i in range(len(route) - 1))

    def route_load(self, route):
        """Tính tổng tải trọng hiện tại của một lộ trình."""
        # Trừ 2 vì loại bỏ node 0 ở đầu và cuối
        return max(0, len(route) - 2) * self.demand

    def total_cost(self, routes):
        """
        HÀM MỤC TIÊU (Objective Function):
        Kết hợp Quãng đường và Hình phạt số lượng xe.
        """
        active_routes = [r for r in routes if len(r) > 2]
        distance = sum(self.route_cost(r) for r in active_routes)
        
        # THAY ĐỔI TẠI ĐÂY: 
        # Giá trị penalty này (2000) là "trọng số" để thuật toán cân bằng giữa:
        # - Giảm số xe (Tiết kiệm chi phí vận hành)
        # - Giảm quãng đường (Tiết kiệm xăng/thời gian)
        vehicle_penalty_value = 2000 
        
        return distance + (len(active_routes) * vehicle_penalty_value)

    def initial_solution(self):
        """
        KHỞI TẠO NGHIỆM BAN ĐẦU (Greedy):
        Gom khách hàng theo thứ tự ngẫu nhiên vào các xe để bắt đầu với số xe hợp lý.
        """
        nodes = list(range(1, self.n))
        random.shuffle(nodes) # Giữ tính ngẫu nhiên để SA khám phá nhiều vùng nghiệm khác nhau
        
        routes = []
        route = [0]
        load = 0
        
        for node in nodes:
            if load + self.demand > self.capacity:
                route.append(0)
                routes.append(route)
                route = [0]
                load = 0
            route.append(node)
            load += self.demand
            
        route.append(0)
        routes.append(route)
        return routes

    def get_neighbor(self, routes):
        """
        TOÁN TỬ LÂN CẬN (Neighbor Operators):
        Thực hiện các thay đổi ngẫu nhiên để tìm nghiệm tốt hơn.
        """
        # Chỉ lấy các lộ trình có khách hàng
        new_routes = [r[:] for r in routes if len(r) > 2]
        if not new_routes: return routes
        
        p = random.random()

        # 1. 2-Opt (50%): Tối ưu đường đi bên trong 1 xe (Làm ngắn quãng đường cực tốt)
        if p < 0.5:
            r_idx = random.randint(0, len(new_routes) - 1)
            r = new_routes[r_idx]
            if len(r) > 4:
                i, j = sorted(random.sample(range(1, len(r) - 1), 2))
                # Đảo ngược đoạn lộ trình để gỡ các đường cắt chéo
                r[i:j] = r[i:j][::-1]

        # 2. Relocate (40%): Di chuyển khách hàng từ xe này sang xe khác (Để gom xe)
        elif p < 0.9:
            r1 = random.randint(0, len(new_routes) - 1)
            if len(new_routes[r1]) > 2:
                # Lấy 1 node ngẫu nhiên ra khỏi xe r1
                node = new_routes[r1].pop(random.randint(1, len(new_routes[r1]) - 2))
                
                # Chọn ngẫu nhiên xe r2 để nhét node vào
                r2 = random.randint(0, len(new_routes) - 1)
                if self.route_load(new_routes[r2]) + self.demand <= self.capacity:
                    new_routes[r2].insert(random.randint(1, len(new_routes[r2]) - 1), node)
                else:
                    # Nếu xe r2 đầy, tạo hẳn một lộ trình mới (giúp linh hoạt số xe)
                    new_routes.append([0, node, 0])

        # 3. Swap (10%): Tráo đổi khách hàng giữa 2 lộ trình khác nhau
        else:
            if len(new_routes) >= 2:
                r1, r2 = random.sample(range(len(new_routes)), 2)
                if len(new_routes[r1]) > 2 and len(new_routes[r2]) > 2:
                    i = random.randint(1, len(new_routes[r1]) - 2)
                    j = random.randint(1, len(new_routes[r2]) - 2)
                    new_routes[r1][i], new_routes[r2][j] = new_routes[r2][j], new_routes[r1][i]

        return [r for r in new_routes if len(r) > 2]

    def solve(self):
        """
        HÀM CHẠY THUẬT TOÁN CHÍNH:
        Mô phỏng quá trình hạ nhiệt để tìm nghiệm tối ưu.
        """
        start_time = time.time()
        
        # Khởi tạo nghiệm hiện tại và nghiệm tốt nhất
        current = self.initial_solution()
        best = [r[:] for r in current]
        
        current_cost = self.total_cost(current)
        best_cost = current_cost
        
        T = self.T_start

        while T > self.T_min:
            # Kiểm tra thời gian chạy tối đa
            if time.time() - start_time > self.max_runtime:
                break
            
            for _ in range(self.iter_per_T):
                # Tạo lân cận
                neighbor = self.get_neighbor(current)
                neighbor_cost = self.total_cost(neighbor)
                
                delta = neighbor_cost - current_cost
                
                # CƠ CHẾ CHẤP NHẬN NGHIỆM CỦA SA (Metropolis Criterion)
                # Chấp nhận nếu tốt hơn, hoặc chấp nhận nghiệm xấu dựa trên xác suất (nhiệt độ)
                if delta < 0 or (T > 0 and random.random() < math.exp(-min(delta/T, 700))):
                    current = neighbor
                    current_cost = neighbor_cost
                    
                    # Cập nhật nghiệm tốt nhất tìm được từ trước đến nay
                    if current_cost < best_cost:
                        best = [r[:] for r in current]
                        best_cost = current_cost
            
            # Hạ nhiệt độ theo hệ số alpha
            T *= self.alpha

        # QUAN TRỌNG: Trả về quãng đường thực tế (không bao gồm penalty số xe)
        actual_dist = sum(self.route_cost(r) for r in best)
        
        # Trả về kết quả cuối cùng
        routes_dict = {i: r for i, r in enumerate(best) if len(r) > 2}
        return routes_dict, actual_dist