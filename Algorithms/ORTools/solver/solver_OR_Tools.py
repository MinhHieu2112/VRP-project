# Lớp bọc thực thi thuật toán tối ưu hóa VRP bằng thư viện OR-Tools của Google.
import math
import time
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


class ORToolsSolver:
    """Bộ giải VRP sử dụng Guided Local Search của thư viện OR-Tools."""

    def __init__(self, data: dict, config: dict):
        # Khởi tạo mô hình định tuyến OR-Tools với các ràng buộc và số lượng xe.
        self.data   = data
        self.config = config

        self.depot_id = data.get(
            "depot",
            config.get("common_model_parameters", {}).get("depot_id", 0)
        )

        n_customers      = len(data["distance_matrix"]) - 1
        vehicle_capacity = data["vehicle_capacity"]
        lb_vehicles      = math.ceil(n_customers / vehicle_capacity)
        max_vehicles     = data["num_vehicles"]
        num_vehicles     = min(int(lb_vehicles * 1.10) + 1, max_vehicles)

        print(f"[ORTools] n_customers={n_customers}, capacity={vehicle_capacity}")
        print(f"[ORTools] Lower-bound xe={lb_vehicles}, "
              f"dùng={num_vehicles} (max={max_vehicles})")

        self._num_vehicles = num_vehicles

        self.manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]),
            num_vehicles,
            self.depot_id
        )
        self.routing = pywrapcp.RoutingModel(self.manager)

    def _distance_callback(self, from_idx: int, to_idx: int) -> int:
        # Callback tính khoảng cách giữa hai node cho solver.
        from_node = self.manager.IndexToNode(from_idx)
        to_node   = self.manager.IndexToNode(to_idx)
        return int(self.data["distance_matrix"][from_node][to_node])

    def _demand_callback(self, from_idx: int) -> int:
        # Callback trả về nhu cầu vận chuyển của một node cho ràng buộc tải trọng.
        return int(self.data["demands"][self.manager.IndexToNode(from_idx)])

    def _make_solution_callback(self, no_improve_limit: int):
        # Tạo callback dừng sớm khi lời giải không cải thiện sau nhiều vòng.
        state = {"best_cost": float("inf"), "no_improve": 0, "calls": 0}

        def callback():
            current = self.routing.CostVar().Min()
            state["calls"] += 1
            if current < state["best_cost"]:
                improvement = state["best_cost"] - current
                state["best_cost"] = current
                state["no_improve"] = 0
                print(f"  [#{state['calls']:>4}] Cải thiện: "
                      f"{current}  (↓{improvement})")
            else:
                state["no_improve"] += 1
                if state["no_improve"] % 20 == 0:
                    print(f"  [#{state['calls']:>4}] Không cải thiện "
                          f"{state['no_improve']}/{no_improve_limit}")
                if state["no_improve"] >= no_improve_limit:
                    print(f"  → Dừng sớm sau {no_improve_limit} vòng.")
                    self.routing.solver().FinishCurrentSearch()

        return callback

    def _extract(self, solution):
        # Trích xuất các tuyến đường từ nghiệm của OR-Tools thành dict kết quả.
        routes         = {}
        total_distance = 0
        vehicle_count  = 0

        for v_id in range(self._num_vehicles):
            index = self.routing.Start(v_id)
            if self.routing.IsEnd(solution.Value(self.routing.NextVar(index))):
                continue

            vehicle_count += 1
            route_nodes    = []

            while not self.routing.IsEnd(index):
                node = self.manager.IndexToNode(index)
                route_nodes.append(node)
                prev_index = index
                index      = solution.Value(self.routing.NextVar(index))
                total_distance += self.data["distance_matrix"][
                    self.manager.IndexToNode(prev_index)
                ][self.manager.IndexToNode(index)]

            route_nodes.append(self.manager.IndexToNode(index))
            routes[vehicle_count] = route_nodes

        return routes, total_distance

    def solve(self, no_improve_iters: int = None):
        # Thực hiện quá trình giải VRP bằng OR-Tools và trả về các tuyến đường tối ưu.
        solver_cfg = self.config.get("solvers", {}).get("or_tools", {})

        transit_cb = self.routing.RegisterTransitCallback(self._distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

        demand_cb = self.routing.RegisterUnaryTransitCallback(self._demand_callback)
        self.routing.AddDimension(
            demand_cb,
            0,
            self.data["vehicle_capacity"],
            True,
            "Capacity"
        )

        no_improve_limit = (
            no_improve_iters
            or solver_cfg.get("no_improve_iters", 200)
        )
        self.routing.AddAtSolutionCallback(
            self._make_solution_callback(no_improve_limit)
        )

        params = pywrapcp.DefaultRoutingSearchParameters()

        fs_str = (
            solver_cfg.get("first_solution_strategy", "PATH_CHEAPEST_ARC")
            .upper().replace(" ", "_")
        )
        try:
            params.first_solution_strategy = getattr(
                routing_enums_pb2.FirstSolutionStrategy, fs_str
            )
        except AttributeError:
            params.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )

        algo_str = (
            solver_cfg.get("local_search_metaheuristic", "GUIDED_LOCAL_SEARCH")
            .upper().replace(" ", "_")
        )
        try:
            params.local_search_metaheuristic = getattr(
                routing_enums_pb2.LocalSearchMetaheuristic, algo_str
            )
        except AttributeError:
            params.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )

        params.time_limit.seconds     = solver_cfg.get("time_limit", 200)
        params.lns_time_limit.seconds = solver_cfg.get("lns_time_per_step_seconds", 1)
        params.log_search             = solver_cfg.get("log_search", False)

        print(f"--- OR-Tools bắt đầu giải ---")
        print(f"    Strategy  : {fs_str}")
        print(f"    Algorithm : {algo_str}")
        print(f"    Dừng sau  : {no_improve_limit} vòng không cải thiện")
        print(f"    Safety net: {solver_cfg.get('time_limit', 200)}s")

        start_time = time.time()
        solution   = self.routing.SolveWithParameters(params)
        solve_time = time.time() - start_time

        if solution:
            routes, total_units = self._extract(solution)
            total_km = total_units / 100.0
            print(f"--- Giải xong trong {solve_time:.2f}s. "
                  f"Tổng quãng đường: {total_km:.2f} km "
                  f"({total_units} units) ---")
            return routes, total_units
        else:
            print("--- Không tìm thấy lời giải! ---")
            return None, None