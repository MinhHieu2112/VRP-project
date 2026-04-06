import time
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

class ORToolsSolver:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        # 1. Lấy thông số từ common_model_parameters
        model_params = config.get("common_model_parameters", {})
        self.depot_id = model_params.get("depot_id", 0)

        # 2. Khởi tạo Manager và Routing Model
        self.manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), 
            data["num_vehicles"], 
            self.depot_id
        )
        self.routing = pywrapcp.RoutingModel(self.manager)

    def _distance_callback(self, from_idx, to_idx):
        """Trả về khoảng cách giữa hai node."""
        from_node = self.manager.IndexToNode(from_idx)
        to_node = self.manager.IndexToNode(to_idx)
        actual_dist = self.data["distance_matrix"][from_node][to_node]
        
        return actual_dist

    def _demand_callback(self, from_idx):
        """Trả về nhu cầu tải trọng tại node."""
        return self.data["demands"][self.manager.IndexToNode(from_idx)]

    def _extract(self, solution):
        routes = {}
        total_distance = 0
        vehicle_count = 0

        for v_id in range(self.data["num_vehicles"]):
            index = self.routing.Start(v_id)

            if self.routing.IsEnd(solution.Value(self.routing.NextVar(index))):
                continue

            vehicle_count += 1
            route_nodes = []

            while not self.routing.IsEnd(index):

                node = self.manager.IndexToNode(index)
                route_nodes.append(node)

                previous_index = index
                index = solution.Value(self.routing.NextVar(index))

                from_node = self.manager.IndexToNode(previous_index)
                to_node = self.manager.IndexToNode(index)

                total_distance += self.data["distance_matrix"][from_node][to_node]

            # thêm depot cuối
            route_nodes.append(self.manager.IndexToNode(index))

            routes[vehicle_count] = route_nodes

        return routes, total_distance

    def solve(self):
        solver_cfg = self.config.get("solvers", {}).get("or_tools", {})
        
        # 1. Đăng ký các hàm Callback
        transit_callback_index = self.routing.RegisterTransitCallback(self._distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(self._demand_callback)
        capacities = [self.config["global_constraints"]["vehicle_capacity"]] * self.data["num_vehicles"]
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, capacities, True, "Capacity"
        )

        # 2. Cấu hình tham số tìm kiếm từ Config
        params = pywrapcp.DefaultRoutingSearchParameters()
        
        # Ánh xạ First Solution Strategy
        fs_str = solver_cfg.get("first_solution_strategy", "PARALLEL_CHEAPEST_INSERTION")
        params.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, fs_str)
        
        # Ánh xạ Metaheuristic (Xử lý khoảng trắng nếu có)
        algo_str = solver_cfg.get("algorithm", "GUIDED_LOCAL_SEARCH").replace(" ", "_")
        params.local_search_metaheuristic = getattr(routing_enums_pb2.LocalSearchMetaheuristic, algo_str)
        
        params.time_limit.seconds = solver_cfg.get("time_limit", 180)
        params.log_search = True 

        # 3. Giải bài toán
        print(f"--- Đang tìm kiếm với: {fs_str} + {algo_str} ---")
        solution = self.routing.SolveWithParameters(params)

        if solution:
            return self._extract(solution)
        return None, 0