import math
import time
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


class ORToolsSolver:
    def __init__(self, data: dict, config: dict):
        """
        Parameters
        ----------
        data   : dict từ DataLoader.load_data() — đã chuẩn hóa theo Pipeline
        config : dict từ Utils/config.json
        """
        self.data   = data
        self.config = config

        # [FIX-2] Ưu tiên data["depot"] (chuẩn Pipeline), fallback config
        self.depot_id = data.get(
            "depot",
            config.get("common_model_parameters", {}).get("depot_id", 0)
        )

        # [FIX-4] Số xe thực tế = lower bound + buffer 10%, tối đa = max_vehicles
        n_customers      = len(data["distance_matrix"]) - 1        # trừ depot
        vehicle_capacity = data["vehicle_capacity"]                 # [FIX-1]
        lb_vehicles      = math.ceil(n_customers / vehicle_capacity)
        max_vehicles     = data["num_vehicles"]                     # từ config
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

    # ── Callbacks ─────────────────────────────────────────────────────────
    def _distance_callback(self, from_idx: int, to_idx: int) -> int:
        from_node = self.manager.IndexToNode(from_idx)
        to_node   = self.manager.IndexToNode(to_idx)
        return int(self.data["distance_matrix"][from_node][to_node])

    def _demand_callback(self, from_idx: int) -> int:
        return int(self.data["demands"][self.manager.IndexToNode(from_idx)])

    def _make_solution_callback(self, no_improve_limit: int):
        """Dừng sớm khi không cải thiện sau `no_improve_limit` lần liên tiếp."""
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

    # ── Extract solution ───────────────────────────────────────────────────
    def _extract(self, solution):
        """
        Trả về (routes_dict, total_distance_units).
        total_distance_units theo đơn vị nội bộ của DataLoader (1 unit = 10m).
        Caller dùng Pipeline.matrix_units_to_km() để đổi sang km.  [FIX-3]
        """
        routes         = {}
        total_distance = 0
        vehicle_count  = 0

        for v_id in range(self._num_vehicles):
            index = self.routing.Start(v_id)
            if self.routing.IsEnd(solution.Value(self.routing.NextVar(index))):
                continue  # xe rỗng

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

            route_nodes.append(self.manager.IndexToNode(index))  # về depot
            routes[vehicle_count] = route_nodes

        return routes, total_distance

    # ── Main solve ─────────────────────────────────────────────────────────
    def solve(self, no_improve_iters: int = None):
        """
        Trả về (routes_dict, total_distance_units).
        Để chuyển sang km: total_km = Pipeline.matrix_units_to_km(total_distance_units)
        """
        solver_cfg = self.config.get("solvers", {}).get("or_tools", {})

        # 1. Arc cost
        transit_cb = self.routing.RegisterTransitCallback(self._distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

        # 2. Capacity dimension  [FIX-1]
        demand_cb = self.routing.RegisterUnaryTransitCallback(self._demand_callback)
        self.routing.AddDimension(
            demand_cb,
            0,
            self.data["vehicle_capacity"],   # lấy từ data, không từ config raw
            True,
            "Capacity"
        )

        # 3. Early-stop callback  [FIX-5]
        no_improve_limit = (
            no_improve_iters
            or solver_cfg.get("no_improve_iters", 200)   # tăng default lên 200
        )
        self.routing.AddAtSolutionCallback(
            self._make_solution_callback(no_improve_limit)
        )

        # 4. Search params
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
        params.lns_time_limit.seconds = solver_cfg.get("lns_time_per_step_seconds", 1)  # [FIX-5]
        params.log_search             = solver_cfg.get("log_search", False)

        print(f"--- OR-Tools bắt đầu giải ---")
        print(f"    Strategy  : {fs_str}")
        print(f"    Algorithm : {algo_str}")
        print(f"    Dừng sau  : {no_improve_limit} vòng không cải thiện")
        print(f"    Safety net: {solver_cfg.get('time_limit', 200)}s")

        # 5. Solve
        start_time = time.time()
        solution   = self.routing.SolveWithParameters(params)
        solve_time = time.time() - start_time

        if solution:
            routes, total_units = self._extract(solution)
            # In ra km để dễ đọc (1 unit = 10m → /100 = km)
            total_km = total_units / 100.0
            print(f"--- Giải xong trong {solve_time:.2f}s. "
                  f"Tổng quãng đường: {total_km:.2f} km "
                  f"({total_units} units) ---")
            return routes, total_units   # trả về units, main.py dùng Pipeline để đổi
        else:
            print("--- Không tìm thấy lời giải! ---")
            return None, None