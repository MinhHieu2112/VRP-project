import time
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

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
        to_node   = self.manager.IndexToNode(to_idx)
        return self.data["distance_matrix"][from_node][to_node]

    def _demand_callback(self, from_idx):
        return self.data["demands"][self.manager.IndexToNode(from_idx)]

    def _make_solution_callback(self, no_improve_limit):
        """
        Trả về closure được gọi mỗi khi OR-Tools tìm thấy solution mới tốt hơn.
        Đếm số lần liên tiếp không cải thiện — nếu vượt ngưỡng thì dừng.
        """
        state = {
            "best_cost":      float("inf"),
            "no_improve_count": 0,
            "call_count":     0,
        }

        def callback():
            current_cost = self.routing.CostVar().Min()
            state["call_count"] += 1

            if current_cost < state["best_cost"]:
                improvement = state["best_cost"] - current_cost
                state["best_cost"] = current_cost
                state["no_improve_count"] = 0
                print(
                    f"  [#{state['call_count']:>4}] Cải thiện: "
                    f"{current_cost:.0f}  (↓{improvement:.0f})"
                )
            else:
                state["no_improve_count"] += 1
                if state["no_improve_count"] % 10 == 0:
                    print(
                        f"  [#{state['call_count']:>4}] Không cải thiện "
                        f"{state['no_improve_count']}/{no_improve_limit} vòng"
                    )

                if state["no_improve_count"] >= no_improve_limit:
                    print(
                        f"  → Dừng sớm sau {no_improve_limit} vòng "
                        f"không cải thiện."
                    )
                    self.routing.solver().FinishCurrentSearch()

        return callback

    def _extract(self, solution):
        routes = {}
        total_distance = 0
        vehicle_count  = 0

        for v_id in range(self.data["num_vehicles"]):
            index = self.routing.Start(v_id)

            if self.routing.IsEnd(solution.Value(self.routing.NextVar(index))):
                continue

            vehicle_count += 1
            route_nodes    = []

            while not self.routing.IsEnd(index):
                node = self.manager.IndexToNode(index)
                route_nodes.append(node)

                previous_index = index
                index          = solution.Value(self.routing.NextVar(index))

                from_node       = self.manager.IndexToNode(previous_index)
                to_node         = self.manager.IndexToNode(index)
                total_distance += self.data["distance_matrix"][from_node][to_node]

            route_nodes.append(self.manager.IndexToNode(index))
            routes[vehicle_count] = route_nodes

        return routes, total_distance

    def solve(self, no_improve_iters=None):
        solver_cfg = self.config.get("solvers", {}).get("or_tools", {})

        # ── 1. Distance & capacity ─────────────────────────────────────
        transit_cb = self.routing.RegisterTransitCallback(self._distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

        demand_cb = self.routing.RegisterUnaryTransitCallback(self._demand_callback)
        self.routing.AddDimension(
            demand_cb,
            0,
            self.config.get("global_constraints", {}).get("vehicle_capacity", 200),
            True,
            "Capacity"
        )

        # ── 2. Đăng ký callback hội tụ ────────────────────────────────
        # Ưu tiên tham số truyền vào, sau đó lấy từ config, mặc định 100
        no_improve_limit = (
            no_improve_iters
            or solver_cfg.get("no_improve_iters", 100)
        )
        callback = self._make_solution_callback(no_improve_limit)
        self.routing.AddAtSolutionCallback(callback)

        # ── 3. Search params ───────────────────────────────────────────
        params = pywrapcp.DefaultRoutingSearchParameters()

        fs_str = solver_cfg.get(
            "first_solution_strategy", "PATH_CHEAPEST_ARC"
        ).upper().replace(" ", "_")
        try:
            params.first_solution_strategy = getattr(
                routing_enums_pb2.FirstSolutionStrategy, fs_str
            )
        except AttributeError:
            params.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )

        algo_str = solver_cfg.get(
            "local_search_metaheuristic", "GUIDED_LOCAL_SEARCH"
        ).upper().replace(" ", "_")
        try:
            params.local_search_metaheuristic = getattr(
                routing_enums_pb2.LocalSearchMetaheuristic, algo_str
            )
        except AttributeError:
            params.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )

        # Safety net: nếu callback không kịp dừng thì time_limit sẽ bắt
        params.time_limit.seconds = solver_cfg.get("time_limit", 600)
        params.lns_time_limit.seconds = solver_cfg.get("lns_time_per_step_seconds", 1)
        params.log_search = solver_cfg.get("log_search", False)

        print(f"--- OR-Tools bắt đầu giải ---")
        print(f"    Strategy  : {fs_str}")
        print(f"    Algorithm : {algo_str}")
        print(f"    Dừng sau  : {no_improve_limit} vòng không cải thiện")
        print(f"    Safety net: {solver_cfg.get('time_limit', 600)}s")

        # ── 4. Solve ───────────────────────────────────────────────────
        start_time = time.time()
        solution   = self.routing.SolveWithParameters(params)
        solve_time = time.time() - start_time

        if solution:
            routes, total_dist = self._extract(solution)
            print(f"--- Giải xong trong {solve_time:.2f}s. "
                  f"Tổng quãng đường: {total_dist:.2f}km ---")
            return routes, total_dist
        else:
            print("--- Không tìm thấy lời giải! ---")
            return None, None