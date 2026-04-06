import time
import random
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

class NoImprovementMonitor(pywrapcp.SearchMonitor):
    def __init__(self, solver, routing, max_time_no_improvement):
        super(NoImprovementMonitor, self).__init__(solver)
        self._routing = routing
        self._max_time = max_time_no_improvement
        self._last_improvement_time = time.time()
        self._best_cost = float('inf')

    def AtSolution(self):
        # Mỗi khi tìm thấy một lời giải khả thi
        current_cost = self._routing.CostVar().Min()
        if current_cost < self._best_cost:
            self._best_cost = current_cost
            self._last_improvement_time = time.time() # Reset đồng hồ khi có cải thiện
        return True

    def CheckLimit(self):
        # Kiểm tra xem đã quá thời hạn chờ đợi chưa
        if time.time() - self._last_improvement_time > self._max_time:
            return True # Dừng tìm kiếm
        return False

class ORToolsSolver:
    def __init__(self, data, config):
        self.data = data
        self.config = config

        model_params = config.get("common_model_parameters", {})
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
        return self.data["distance_matrix"][from_node][to_node]

    def _demand_callback(self, from_idx):
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

            route_nodes.append(self.manager.IndexToNode(index))
            routes[vehicle_count] = route_nodes

        return routes, total_distance

    

    def solve(self, no_improve_iters=None):
        """
        Giải bài toán bằng OR-Tools với cơ chế ngắt khi không cải thiện.
        """
        solver_cfg = self.config.get("solvers", {}).get("or_tools", {})

        # 1. Thiết lập khoảng cách và ràng buộc (giữ nguyên logic cũ)
        transit_callback_index = self.routing.RegisterTransitCallback(self._distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(self._demand_callback)
        self.routing.AddDimension(
            demand_callback_index,
            0,  # null capacity slack
            self.config.get("global_constraints", {}).get("vehicle_capacity", 200),
            True,  # start cumul to zero
            "Capacity"
        )

        # 2. Cấu hình tham số giải
        params = pywrapcp.DefaultRoutingSearchParameters()
        
        # Lấy chiến lược từ config và chuẩn hóa chuỗi
        fs_str = solver_cfg.get("first_solution_strategy", "PARALLEL_CHEAPEST_INSERTION").upper().replace(" ", "_")
        try:
            params.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, fs_str)
        except AttributeError:
            params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

        algo_str = solver_cfg.get("local_search_metaheuristic", "GUIDED_LOCAL_SEARCH").upper().replace(" ", "_")
        try:
            params.local_search_metaheuristic = getattr(routing_enums_pb2.LocalSearchMetaheuristic, algo_str)
        except AttributeError:
            params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

        # 3. THIẾT LẬP CƠ CHẾ DỪNG (FIX)
        # Vì OR-Tools không có iterations giống PyVRP, ta quy đổi no_improve_iters thành giây chờ đợi.
        # Ví dụ: 40,000 iters trong PyVRP mất ~30-60s, ta đặt ngưỡng 60s không cải thiện cho OR-Tools.
        max_wait_seconds = solver_cfg.get("max_wait_seconds", 60)
        
        # Đăng ký Monitor vào Solver
        improvement_monitor = NoImprovementMonitor(self.routing.solver(), self.routing, max_wait_seconds)
        self.routing.AddSearchMonitor(improvement_monitor)

        # Giới hạn thời gian của mỗi bước LNS (giữ nguyên)
        params.lns_time_limit.seconds = solver_cfg.get("lns_time_per_step_seconds", 5)
        params.log_search = True  # Hiển thị log để bạn theo dõi

        print(f"--- OR-Tools bắt đầu giải ---")
        print(f"--- Chế độ: Dừng sau {max_wait_seconds}s không thấy cải thiện ---")

        # 4. Thực thi giải
        start_time = time.time()
        solution = self.routing.SolveWithParameters(params)
        solve_time = time.time() - start_time

        if solution:
            routes, total_dist = self._extract(solution)
            print(f"--- Giải xong trong {solve_time:.2f}s. Tổng quãng đường: {total_dist:.2f}km ---")
            return routes, total_dist
        else:
            print("--- Không tìm thấy lời giải! ---")
            return None, None