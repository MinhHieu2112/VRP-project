import time
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

class ORToolsSolver:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        
        model_params = config.get("model_parameters", {})
        self.scaling_factor = model_params.get("scaling_factor", 100)
        self.depot_id = model_params.get("depot_id", 0)

        self.manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), 
            data["num_vehicles"], 
            self.depot_id
        )
        self.routing = pywrapcp.RoutingModel(self.manager)

    def _distance_callback(self, from_idx, to_idx):
        from_node = self.manager.IndexToNode(from_idx)
        to_node = self.manager.IndexToNode(to_idx)
        actual_dist = self.data["distance_matrix"][from_node][to_node]
        return int(round(actual_dist * self.scaling_factor))

    def _demand_callback(self, from_idx):
        return self.data["demands"][self.manager.IndexToNode(from_idx)]

    def solve(self):
        solver_cfg = self.config.get("solver_parameters", {})
        
        # Đăng ký callback và thiết lập chi phí
        transit_callback_index = self.routing.RegisterTransitCallback(self._distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Thêm ràng buộc tải trọng
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(self._demand_callback)
        capacities = [self.config["constraints"]["vehicle_capacity"]] * self.data["num_vehicles"]
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, capacities, True, "Capacity"
        )

        params = pywrapcp.DefaultRoutingSearchParameters()
        fs_str = solver_cfg.get("first_solution_strategy", "PATH_CHEAPEST_ARC")
        params.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, fs_str)
        
        algo_str = solver_cfg.get("algorithm", "GUIDED_LOCAL_SEARCH").replace(" ", "_")
        params.local_search_metaheuristic = getattr(routing_enums_pb2.LocalSearchMetaheuristic, algo_str)
        
        params.time_limit.seconds = solver_cfg.get("time_limit", 180)
        params.log_search = True 

        solution = self.routing.SolveWithParameters(params)

        if solution:
            return self._extract(solution)
        return None, 0

    def _extract(self, solution):
        """
        Trích xuất lộ trình và tính toán 'True Distance' từ ma trận gốc.
        Giải quyết vấn đề chênh lệch do ArcCost và thiếu chặng Depot cuối.
        """
        routes = {}
        true_total_distance = 0.0
        active_vehicles_count = 0
        
        # Truy xuất trực tiếp ma trận khoảng cách thực (km)
        dist_matrix = self.data["distance_matrix"]

        for v_id in range(self.data["num_vehicles"]):
            index = self.routing.Start(v_id)
            
            # Kiểm tra nếu xe không hoạt động (đi từ Depot thẳng đến Depot)
            if self.routing.IsEnd(solution.Value(self.routing.NextVar(index))):
                continue
            
            active_vehicles_count += 1
            route_nodes = []
            
            while True:
                node_index = self.manager.IndexToNode(index)
                route_nodes.append(node_index)
                
                if self.routing.IsEnd(index):
                    break
                
                # Tính toán khoảng cách thực tế giữa node hiện tại và node kế tiếp
                prev_node = node_index
                next_index = solution.Value(self.routing.NextVar(index))
                next_node = self.manager.IndexToNode(next_index)
                
                # Cộng dồn từ ma trận gốc (km), không dùng GetArcCost
                true_total_distance += dist_matrix[prev_node][next_node]
                index = next_index

            # Lưu lộ trình (đã bao gồm Depot đầu 0 và Depot cuối 0)
            routes[active_vehicles_count] = route_nodes

        return routes, true_total_distance