"""
Algorithms/ACO/Core/engine.py  (refactored)
============================================
BasicACO — thuật toán Ant Colony Optimization cho CVRP.

Sửa lỗi so với phiên bản cũ:
  [FIX-1] select_next_index: feasible_nodes là list tùy ý, không phải
          index liên tục → dùng np.array fancy indexing nhất quán.
  [FIX-2] _construct_solution: khi force_visit_remaining xảy ra, vẫn
          gọi local_update_pheromone cho cạnh cuối về depot.
  [FIX-3] _count_valid_vehicles: trường hợp path kết thúc không phải
          depot vẫn được đếm đúng.
"""

import numpy as np
import random
import time
from Models.cvrp_base import CVRPGraph
from Core.ant import Ant


class BasicACO:
    def __init__(self,
                 graph: CVRPGraph,
                 ants_num: int = 10,
                 max_iter: int = 50_000,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 q0: float = 0.9,
                 no_improve_limit: int = 1000):
        """
        Khởi tạo ACO.

        Parameters
        ----------
        graph            : Đồ thị CVRP (chứa pheromone, heuristic, dist).
        ants_num         : Số kiến mỗi vòng lặp.
        max_iter         : Số vòng lặp tối đa (safety cap).
        alpha            : Trọng số pheromone (thường = 1).
        beta             : Trọng số heuristic (thường = 2–5).
        q0               : Xác suất khai thác (exploitation) theo ACS.
        no_improve_limit : Dừng sớm sau n vòng không cải thiện best.
        """
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

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def run_basic_aco(self):
        """Chạy ACO và trả về (best_path, best_distance, best_vehicle_num)."""
        self._basic_aco()
        return self.best_path, self.best_path_distance, self.best_vehicle_num

    # ──────────────────────────────────────────────────────────────────
    # Core loop
    # ──────────────────────────────────────────────────────────────────

    def _basic_aco(self):
        """Vòng lặp ACO chính: xây nghiệm → cập nhật best → update pheromone."""
        start_time       = time.time()
        no_improve_count = 0

        for iteration in range(self.max_iter):

            if no_improve_count >= self.no_improve_limit:
                print(f"[ACO] Dừng sớm tại vòng {iteration}: "
                      f"{no_improve_count} vòng không cải thiện "
                      f"(ngưỡng={self.no_improve_limit}).")
                break

            ants = [Ant(self.graph) for _ in range(self.ants_num)]

            for ant in ants:
                self._construct_solution(ant)

            improved = False
            for ant in ants:
                if (self.best_path is None
                        or ant.total_travel_distance < self.best_path_distance):
                    self.best_path          = ant.travel_path[:]
                    self.best_path_distance = ant.total_travel_distance
                    self.best_vehicle_num   = self._count_valid_vehicles(
                        self.best_path)
                    improved = True
                    print(f"[iter {iteration:5d}] Improved: "
                          f"dist={self.best_path_distance:.0f}m "
                          f"({self.best_path_distance / 1000:.2f}km), "
                          f"vehicles={self.best_vehicle_num}")

            no_improve_count = 0 if improved else no_improve_count + 1

            self.graph.global_update_pheromone(
                self.best_path, self.best_path_distance)

            if iteration % 50 == 0 and iteration > 0:
                elapsed = time.time() - start_time
                print(f"  [iter {iteration:5d}] elapsed={elapsed:.1f}s | "
                      f"NoImprove={no_improve_count}/{self.no_improve_limit}")

        elapsed = time.time() - start_time
        print(f"[DONE] dist={self.best_path_distance:.0f}m "
              f"({self.best_path_distance / 1000:.2f}km), "
              f"vehicles={self.best_vehicle_num}, "
              f"time={elapsed:.2f}s")

    # ──────────────────────────────────────────────────────────────────
    # Xây nghiệm cho một kiến
    # ──────────────────────────────────────────────────────────────────

    def _construct_solution(self, ant: Ant):
        """
        Kiến xây lộ trình hoàn chỉnh từ depot.
        Khi không còn node khả thi → về depot (mở tuyến mới).
        Khi vượt max_steps → force-visit còn lại (giữ capacity constraint).
        """
        n_customers = self.graph.node_num - 1
        max_steps   = 2 * n_customers + 10  # tối đa 2 lần số khách (depot returns)

        steps = 0
        while not ant.index_to_visit_empty():
            steps += 1

            if steps > max_steps:
                remaining = ant.index_to_visit  # snapshot sorted list
                print(f"[WARN] max_steps vượt quá ({max_steps}), "
                      f"force-visit {len(remaining)} node còn lại")
                ant.force_visit_remaining(remaining)
                break

            feasible = ant.cal_next_index_meet_constrains()

            if not feasible:
                # Không có node nào vừa capacity → về depot mở tuyến mới
                if not ant.is_at_depot():
                    prev = ant.current_index
                    ant.move_to_next_index(0)
                    self.graph.local_update_pheromone(prev, 0)
                else:
                    # Kẹt tại depot (demand đơn lẻ > capacity) → force visit
                    print("[ERROR] Ant kẹt tại depot — dữ liệu có node demand > capacity.")
                    ant.force_visit_remaining(ant.index_to_visit)
                    break
            else:
                prev       = ant.current_index
                next_index = self.select_next_index(ant, feasible)
                ant.move_to_next_index(next_index)
                self.graph.local_update_pheromone(prev, next_index)

        # Đảm bảo kết thúc tại depot
        if not ant.is_at_depot():
            prev = ant.current_index
            ant.move_to_next_index(0)
            # [FIX-2] Cập nhật pheromone cạnh cuối về depot
            self.graph.local_update_pheromone(prev, 0)

    # ──────────────────────────────────────────────────────────────────
    # Chọn node tiếp theo (ACS rule)
    # ──────────────────────────────────────────────────────────────────

    def select_next_index(self, ant: Ant, feasible_nodes: list) -> int:
        """
        Chọn node tiếp theo theo quy tắc ACS:
          - Với xác suất q0: chọn node có score cao nhất (exploitation).
          - Ngược lại: chọn theo roulette wheel (exploration).
        """
        current_index = ant.current_index

        # [FIX-1] Dùng np.array để fancy indexing ổn định
        feasible_arr = np.array(feasible_nodes, dtype=int)
        pheromone    = self.graph.pheromone_mat[current_index][feasible_arr]
        heuristic    = self.graph.heuristic_info_mat[current_index][feasible_arr]

        if self.alpha == 1:
            scores = pheromone * np.power(heuristic, self.beta)
        else:
            scores = np.power(pheromone, self.alpha) * np.power(heuristic, self.beta)

        scores    = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        score_sum = scores.sum()

        if score_sum <= 0:
            return random.choice(feasible_nodes)

        if random.random() < self.q0:
            best_local_idx = int(np.argmax(scores))
            return int(feasible_arr[best_local_idx])
        else:
            probs = scores / score_sum
            return self._roulette_wheel(feasible_nodes, probs)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _roulette_wheel(candidates: list, probs: np.ndarray) -> int:
        """Chọn node theo xác suất roulette wheel (exploration)."""
        probs = np.clip(probs, 0, None)
        total = probs.sum()
        if total <= 0:
            return random.choice(candidates)
        probs      = probs / total
        chosen_idx = np.random.choice(len(candidates), p=probs)
        return candidates[chosen_idx]

    @staticmethod
    def _count_valid_vehicles(path: list) -> int:
        """
        Đếm số xe thực sự sử dụng trong path.
        Một xe được tính khi route có ít nhất 1 khách hàng (node != 0).

        [FIX-3] Xử lý đúng khi path không kết thúc bằng depot.
        """
        count        = 0
        has_customer = False

        for node in path[1:]:
            if node == 0:
                if has_customer:
                    count += 1
                has_customer = False
            else:
                has_customer = True

        # Trường hợp path kết thúc không phải depot
        if has_customer:
            count += 1

        return count