import time
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

class ORToolsSolver:
    def __init__(self, data):
        self.data = data
        self.manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
        self.routing = pywrapcp.RoutingModel(self.manager)

    def _distance_callback(self, from_idx, to_idx):
        return self.data["distance_matrix"][self.manager.IndexToNode(from_idx)][self.manager.IndexToNode(to_idx)]

    def _demand_callback(self, from_idx):
        return self.data["demands"][self.manager.IndexToNode(from_idx)]

    def solve(self, time_limit=30):
        t_idx = self.routing.RegisterTransitCallback(self._distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(t_idx)
        
        d_idx = self.routing.RegisterUnaryTransitCallback(self._demand_callback)
        self.routing.AddDimensionWithVehicleCapacity(d_idx, 0, self.data["vehicle_capacities"], True, "Capacity")

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.seconds = time_limit

        start_time = time.time()
        solution = self.routing.SolveWithParameters(params)
        duration = time.time() - start_time

        if solution:
            # LƯU Ý: Gọi hàm extract và trả về kết quả của nó
            return self._extract(solution, duration)
        else:
            print(" Không tìm thấy lời giải (Có thể do Time Limit hoặc Capacity)")
            # SỬA Ở ĐÂY: Chỉ trả về 2 giá trị None, 0 (Thay vì None, 0, duration)
            return None, 0

    def _extract(self, solution, duration):
        routes = {}
        total_dist_scaled = 0
        active_vehicles = 0
        
        print("\n" + "="*50)
        print(f" CHI TIẾT LỘ TRÌNH (Thời gian giải: {duration:.2f}s)")
        print("="*50)

        for v_id in range(self.data["num_vehicles"]):
            index = self.routing.Start(v_id)
            
            # Kiểm tra xem xe này có thực sự di chuyển không
            if self.routing.IsEnd(solution.Value(self.routing.NextVar(index))):
                continue
            
            active_vehicles += 1
            route_nodes = []
            route_distance = 0
            
            # Duyệt qua các điểm trong lộ trình của xe v_id
            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                route_nodes.append(node_index)
                
                previous_index = index
                index = solution.Value(self.routing.NextVar(index))
                
                # Cộng dồn khoảng cách (đã scale)
                route_distance += self.routing.GetArcCostForVehicle(previous_index, index, v_id)

            # Quay về Depot
            route_nodes.append(self.data["depot"])
            
            # Lưu vào dictionary để trả về (dùng cho Visualizer)
            routes[f"{active_vehicles:03d}"] = route_nodes
            total_dist_scaled += route_distance

            # --- IN TUYẾN ĐƯỜNG RA MÀN HÌNH ---
            dist_km = route_distance / self.data["scaling_factor"]
            print(f" Xe #{active_vehicles:03d}:")
            # Nối các điểm bằng dấu mũi tên
            path_str = " -> ".join(map(str, route_nodes))
            print(f"   Lộ trình: {path_str}")
            print(f"   Quãng đường: {dist_km:.2f} km")
            print("-" * 20)

        final_distance = total_dist_scaled / self.data["scaling_factor"]
        
        print("="*50)
        print(f" TỔNG KẾT:")
        print(f"   - Tổng quãng đường: {final_distance:.2f} km")
        print(f"   - Số xe sử dụng: {active_vehicles}/{self.data['num_vehicles']}")
        print("="*50 + "\n")

        return routes, final_distance