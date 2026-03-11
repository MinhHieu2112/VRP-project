import time
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

class ORToolsSolver:
    def __init__(self, data):
        self.data = data
        # Manager giúp quản lý các chỉ số node và phương tiện
        self.manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), 
            data["num_vehicles"], 
            data["depot"]
        )
        self.routing = pywrapcp.RoutingModel(self.manager)

    def _distance_callback(self, from_idx, to_idx):
        return self.data["distance_matrix"][self.manager.IndexToNode(from_idx)][self.manager.IndexToNode(to_idx)]

    def _demand_callback(self, from_idx):
        return self.data["demands"][self.manager.IndexToNode(from_idx)]

    def solve(self, time_limit=300):
        # 1. Đăng ký hàm tính khoảng cách
        t_idx = self.routing.RegisterTransitCallback(self._distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(t_idx)
        
        # 2. Đăng ký hàm tính tải trọng (Capacity)
        d_idx = self.routing.RegisterUnaryTransitCallback(self._demand_callback)
        
        # Tạo danh sách sức chứa cho tất cả các xe từ tham số đơn lẻ
        capacities = [self.data["vehicle_capacity"]] * self.data["num_vehicles"]
        
        self.routing.AddDimensionWithVehicleCapacity(
            d_idx, 
            0, # Không có slack (độ trễ tải trọng)
            capacities, 
            True, # Bắt đầu xe với tải trọng 0
            "Capacity"
        )

        # 3. Cấu hình tham số tìm kiếm
        params = pywrapcp.DefaultRoutingSearchParameters()
        # Chiến lược tìm lời giải đầu tiên
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        # Chiến lược tối ưu hóa (Metaheuristic)
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.seconds = time_limit
        # (Tùy chọn) Bật log để xem quá trình nhảy số của OR-Tools
        # params.log_search = True 

        print(f"--- Đang tìm kiếm lời giải với Guided Local Search... ---")
        start_time = time.time()
        solution = self.routing.SolveWithParameters(params)
        duration = time.time() - start_time

        if solution:
            return self._extract(solution, duration)
        else:
            return None, 0

    def _extract(self, solution, duration):
        routes = {}
        total_dist_scaled = 0
        active_vehicles_count = 0
        
        for v_id in range(self.data["num_vehicles"]):
            index = self.routing.Start(v_id)
            
            # Nếu xe chỉ đi từ Depot đến Depot thì bỏ qua
            if self.routing.IsEnd(solution.Value(self.routing.NextVar(index))):
                continue
            
            active_vehicles_count += 1
            route_nodes = []
            
            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                route_nodes.append(node_index)
                previous_index = index
                index = solution.Value(self.routing.NextVar(index))
                total_dist_scaled += self.routing.GetArcCostForVehicle(previous_index, index, v_id)

            route_nodes.append(self.data["depot"])
            # Lưu ID xe dưới dạng số nguyên để main.py dễ xử lý
            routes[active_vehicles_count] = route_nodes

        return routes, total_dist_scaled