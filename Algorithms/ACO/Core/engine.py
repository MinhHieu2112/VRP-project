"""
Algorithms/ACO/Core/engine.py
==============================
BasicACO — Ant Colony System (ACS) cho ACVRP.
Tham chiếu: Dorigo & Gambardella (1997), báo cáo mục 2.4.3.

FIXES:
  [FIX-UNIT]    In đơn vị nhất quán: X units (Y km) với Y=X/100.
                Cũ in /1000 → sai 10× (779 km hiện thành 77.9 km).

  [FIX-FEASIBLE] cal_next_index_meet_constrains() là O(n) Python loop.
                Với 1600 điểm avg remaining ~800 → bottleneck chính.
                Fix: vectorized numpy feasibility check.

  [FIX-EXPLORE] seed_weight=2.0 + q0=0.9 khóa kiến vào greedy path.
                Fix: Exploration boost khi stuck no_improve/2 iterations
                (giảm q0 tạm thời từ 0.9 xuống ~0.7).

  [FIX-1] numpy fancy indexing ổn định (feasible_nodes list tùy ý).
  [FIX-2] local_update cho cạnh cuối về depot sau force_visit.
  [FIX-3] _count_valid_vehicles xử lý path không kết thúc bằng depot.
  [FIX-6] Đếm customer_steps thay total_steps tránh false WARN.
"""

import numpy as np
import random
import time
from Models.cvrp_base import CVRPGraph
from Core.ant import Ant

KM_SCALE = 100  # 1 unit = 10m, 1 km = 100 units


class BasicACO:
    def __init__(self,
                 graph: CVRPGraph,
                 ants_num: int         = 10,
                 max_iter: int         = 50_000,
                 alpha: float          = 1.0,
                 beta: float           = 2.0,
                 q0: float             = 0.9,
                 no_improve_limit: int = 500):
        self.graph            = graph
        self.ants_num         = ants_num
        self.max_iter         = max_iter
        self.alpha            = alpha
        self.beta             = beta
        self.q0               = q0
        self.no_improve_limit = no_improve_limit

        self.best_path_distance: float | None = None
        self.best_path: list | None           = None
        self.best_vehicle_num: int | None     = None

        # [FIX-FEASIBLE] Cache demands array để vectorized check
        self._demands = np.array(
            [graph.nodes[i].demand for i in range(graph.node_num)],
            dtype=np.float64
        )

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def run_basic_aco(self):
        """Chạy ACS, trả về (best_path, best_distance, best_vehicle_num)."""
        self._basic_aco()
        self.best_vehicle_num = self._count_valid_vehicles(self.best_path)
        return self.best_path, self.best_path_distance, self.best_vehicle_num

    # ──────────────────────────────────────────────────────────────────
    # Core loop
    # ──────────────────────────────────────────────────────────────────

    def _basic_aco(self):
        """
        Vòng lặp ACS:
          1. Kiến xây nghiệm + local pheromone update
          2. Cập nhật best
          3. Global pheromone update (1 chiều, ACVRP)
          4. Early stopping

        [FIX-EXPLORE] q0 động: khi stuck no_improve/2 → giảm q0 để explore.
        """
        start_time       = time.time()
        no_improve_count = 0
        q0_current       = self.q0
        half_limit       = max(self.no_improve_limit // 2, 50)

        for iteration in range(self.max_iter):

            if no_improve_count >= self.no_improve_limit:
                print(f"[ACO] Dừng sớm tại vòng {iteration}: "
                      f"{no_improve_count} vòng không cải thiện "
                      f"(ngưỡng={self.no_improve_limit}).")
                break

            # [FIX-EXPLORE] Boost exploration khi stuck giữa chừng
            if no_improve_count == half_limit:
                q0_current = max(self.q0 - 0.2, 0.5)
                print(f"  [ACO] Exploration boost tại iter {iteration}: "
                      f"q0 {self.q0:.2f}→{q0_current:.2f}")
            elif no_improve_count == 0 and q0_current != self.q0:
                q0_current = self.q0  # Khôi phục sau khi improve

            ants = [Ant(self.graph) for _ in range(self.ants_num)]
            for ant in ants:
                self._construct_solution(ant, q0_current)

            improved = False
            for ant in ants:
                if (self.best_path is None
                        or ant.total_travel_distance < self.best_path_distance):
                    self.best_path          = ant.travel_path[:]
                    self.best_path_distance = ant.total_travel_distance
                    self.best_vehicle_num   = self._count_valid_vehicles(
                        self.best_path)
                    improved = True
                    # [FIX-UNIT] In đúng đơn vị: units và km
                    km = self.best_path_distance / KM_SCALE
                    print(f"[iter {iteration:5d}] Improved: "
                          f"{self.best_path_distance:.0f} units ({km:.2f} km), "
                          f"vehicles={self.best_vehicle_num}")

            no_improve_count = 0 if improved else no_improve_count + 1

            self.graph.global_update_pheromone(
                self.best_path, self.best_path_distance)

            if iteration % 50 == 0 and iteration > 0:
                elapsed = time.time() - start_time
                km = self.best_path_distance / KM_SCALE
                print(f"  [iter {iteration:5d}] elapsed={elapsed:.1f}s | "
                      f"NoImprove={no_improve_count}/{self.no_improve_limit} | "
                      f"best={km:.2f} km | q0_eff={q0_current:.2f}")

        elapsed = time.time() - start_time
        km = self.best_path_distance / KM_SCALE
        print(f"\n[DONE] {self.best_path_distance:.0f} units ({km:.2f} km), "
              f"vehicles={self.best_vehicle_num}, time={elapsed:.2f}s")

    # ──────────────────────────────────────────────────────────────────
    # Xây nghiệm cho một kiến
    # ──────────────────────────────────────────────────────────────────

    def _construct_solution(self, ant: Ant, q0: float):
        """
        [FIX-6] Đếm customer_steps (không đếm depot return) → tránh false WARN.
        [FIX-FEASIBLE] Dùng _get_feasible_vectorized thay Python loop.
        [FIX-2] local_update cho cạnh cuối về depot.
        """
        n_customers        = self.graph.node_num - 1
        max_customer_steps = n_customers
        customer_steps     = 0

        while not ant.index_to_visit_empty():

            if customer_steps > max_customer_steps:
                remaining = ant.index_to_visit
                print(f"[WARN] customer_steps ({customer_steps}) > max ({max_customer_steps}), "
                      f"force-visit {len(remaining)} node còn lại")
                ant.force_visit_remaining(remaining)
                break

            # [FIX-FEASIBLE] Vectorized
            feasible = self._get_feasible_vectorized(ant)

            if not feasible:
                if not ant.is_at_depot():
                    prev = ant.current_index
                    ant.move_to_next_index(0)
                    self.graph.local_update_pheromone(prev, 0)
                else:
                    print("[ERROR] Ant kẹt tại depot — demand đơn lẻ > capacity.")
                    ant.force_visit_remaining(ant.index_to_visit)
                    break
            else:
                prev       = ant.current_index
                next_index = self.select_next_index(ant, feasible, q0)
                ant.move_to_next_index(next_index)
                self.graph.local_update_pheromone(prev, next_index)
                if next_index != 0:
                    customer_steps += 1

        if not ant.is_at_depot():
            prev = ant.current_index
            ant.move_to_next_index(0)
            # [FIX-2] Cập nhật pheromone cạnh cuối về depot
            self.graph.local_update_pheromone(prev, 0)

    def _get_feasible_vectorized(self, ant: Ant) -> list:
        """
        [FIX-FEASIBLE] Vectorized feasibility check.

        Thay: [i for i in to_visit if load + demand[i] <= cap]  ← O(n) Python
        Bằng: numpy boolean mask → O(n) C backend, ~5-10× nhanh hơn.

        avg remaining cho 1600 điểm = ~800 nodes:
          Python: 800 × (dict lookup + comparison) ≈ 800 ops, slow
          Numpy:  1 slice + 1 broadcast comparison → 800 ops, fast C
        """
        to_visit_arr = np.array(ant.index_to_visit, dtype=int)
        if len(to_visit_arr) == 0:
            return []

        node_demands  = self._demands[to_visit_arr]
        feasible_mask = (ant.vehicle_load + node_demands) <= self.graph.vehicle_capacity
        return to_visit_arr[feasible_mask].tolist()

    # ──────────────────────────────────────────────────────────────────
    # Chọn node tiếp theo — ACS transition rule
    # ──────────────────────────────────────────────────────────────────

    def select_next_index(self, ant: Ant, feasible_nodes: list, q0: float) -> int:
        """
        ACS rule với q0 động:
          q < q0  → exploitation: argmax score
          q ≥ q0  → exploration: roulette wheel

        [FIX-1] numpy fancy indexing ổn định.
        """
        current_index = ant.current_index
        feasible_arr  = np.array(feasible_nodes, dtype=int)

        pheromone = self.graph.pheromone_mat[current_index][feasible_arr]
        heuristic = self.graph.heuristic_info_mat[current_index][feasible_arr]

        scores = (pheromone if self.alpha == 1.0
                  else np.power(pheromone, self.alpha))
        scores = scores * np.power(heuristic, self.beta)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        score_sum = scores.sum()
        if score_sum <= 0:
            return random.choice(feasible_nodes)

        if random.random() < q0:
            return int(feasible_arr[int(np.argmax(scores))])
        else:
            probs = scores / score_sum
            return self._roulette_wheel(feasible_nodes, probs)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _roulette_wheel(candidates: list, probs: np.ndarray) -> int:
        probs = np.clip(probs, 0, None)
        total = probs.sum()
        if total <= 0:
            return random.choice(candidates)
        chosen_idx = np.random.choice(len(candidates), p=probs / total)
        return candidates[chosen_idx]

    @staticmethod
    def _count_valid_vehicles(path: list) -> int:
        """[FIX-3] Đếm đúng số xe kể cả path không kết thúc bằng depot."""
        count, has_customer = 0, False
        for node in path[1:]:
            if node == 0:
                if has_customer:
                    count += 1
                has_customer = False
            else:
                has_customer = True
        if has_customer:
            count += 1
        return count