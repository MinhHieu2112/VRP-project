import numpy as np
import random
import time
from cvrp_base import CVRPGraph
from ant import Ant


class BasicACO:
    def __init__(self, graph: CVRPGraph,
                 ants_num=10,
                 max_iter=200,
                 alpha=1,       # [FIX C1] Thêm tham số alpha (pheromone importance)
                 beta=2,        # Heuristic importance
                 q0=0.9,        # [FIX M3] ACS gốc dùng q0 ≈ 0.9 (exploitation > exploration)
                 no_improve_limit=50):
        self.graph = graph
        self.ants_num = ants_num
        self.max_iter = max_iter
        self.alpha = alpha      # [FIX C1]
        self.beta = beta
        self.q0 = q0            # [FIX M3]
        self.no_improve_limit = no_improve_limit

        self.best_path_distance = None
        self.best_path = None
        self.best_vehicle_num = None

    def run_basic_aco(self):
        """[FIX S5] Bỏ Thread wrapper vô nghĩa, gọi trực tiếp."""
        self._basic_aco()
        return self.best_path, self.best_path_distance, self.best_vehicle_num

    def _basic_aco(self):
        start_time = time.time()
        no_improve_count = 0

        for iteration in range(self.max_iter):
            ants = [Ant(self.graph) for _ in range(self.ants_num)]

            for ant in ants:
                self._construct_solution(ant)

            # Cập nhật best solution
            improved = False
            for ant in ants:
                if (self.best_path is None
                        or ant.total_travel_distance < self.best_path_distance):
                    self.best_path = ant.travel_path[:]
                    self.best_path_distance = ant.total_travel_distance
                    # [FIX S4] Đếm vehicle đúng: loại bỏ route rỗng
                    self.best_vehicle_num = self._count_valid_vehicles(self.best_path)
                    improved = True
                    print(f'[iter {iteration}] Improved: dist={self.best_path_distance:.2f}, '
                          f'vehicles={self.best_vehicle_num}')

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1

            self.graph.global_update_pheromone(self.best_path, self.best_path_distance)

            if no_improve_count >= self.no_improve_limit:
                print(f'Early stop tại iteration {iteration} (no improvement for {self.no_improve_limit} iters)')
                break

        print(f'[DONE] dist={self.best_path_distance:.2f}, '
              f'vehicles={self.best_vehicle_num}, '
              f'time={time.time()-start_time:.2f}s')

    def _construct_solution(self, ant: Ant):
        """
        [FIX C3] Xây dựng solution với bảo vệ chống dead-end và vòng lặp vô tận.
        """
        max_steps = self.graph.node_num * 3  # Giới hạn an toàn
        steps = 0

        while not ant.index_to_visit_empty():
            steps += 1
            if steps > max_steps:
                # Fallback: force thăm remaining nodes bất kể capacity
                # (Tạo infeasible solution, sẽ bị dominated bởi feasible solutions)
                for remaining in ant.index_to_visit:
                    if ant.current_index != 0:
                        ant.move_to_next_index(0)
                        self.graph.local_update_pheromone(ant.current_index, 0)
                    ant.move_to_next_index(remaining)
                    self.graph.local_update_pheromone(ant.current_index, remaining)
                break

            feasible = ant.cal_next_index_meet_constrains()

            if not feasible:
                # [FIX C3] Không còn node nào feasible → quay về depot
                # Nhưng tránh depot→depot liên tiếp [FIX S4]
                if not ant.is_at_depot():
                    ant.move_to_next_index(0)
                    self.graph.local_update_pheromone(ant.current_index, 0)
                else:
                    # Đang ở depot mà không có node feasible nào
                    # → Có thể data lỗi (demand > capacity). Break tránh loop vô tận.
                    print('[WARNING] Ant bị kẹt tại depot, không có node feasible. Kiểm tra data.')
                    break
            else:
                next_index = self.select_next_index(ant, feasible)
                ant.move_to_next_index(next_index)
                self.graph.local_update_pheromone(ant.current_index, next_index)

        # Kết thúc: quay về depot nếu chưa ở đó
        if not ant.is_at_depot():
            ant.move_to_next_index(0)
            self.graph.local_update_pheromone(ant.current_index, 0)

    def select_next_index(self, ant: Ant, feasible_nodes: list) -> int:
        """
        [FIX C1] Dùng alpha trong công thức transition probability.
        [FIX C2] feasible_nodes được truyền vào rõ ràng, không tái tính.
        """
        current_index = ant.current_index

        # Tính transition probability với alpha và beta
        pheromone = self.graph.pheromone_mat[current_index][feasible_nodes]
        heuristic = self.graph.heuristic_info_mat[current_index][feasible_nodes]

        # [FIX C1] Công thức đầy đủ: τ^α · η^β
        scores = (np.power(pheromone, self.alpha)
                  * np.power(heuristic, self.beta))

        # Làm sạch NaN/Inf
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        score_sum = scores.sum()
        if score_sum <= 0:
            # Fallback uniform nếu tất cả score = 0
            return random.choice(feasible_nodes)

        # ACS: q0 → exploitation (chọn max), ngược lại → stochastic
        if random.random() < self.q0:
            # Exploitation: chọn node có score cao nhất
            best_local_idx = int(np.argmax(scores))
            return feasible_nodes[best_local_idx]
        else:
            # Exploration: roulette wheel
            probs = scores / score_sum
            return self._roulette_wheel(feasible_nodes, probs)

    @staticmethod
    def _roulette_wheel(candidates: list, probs: np.ndarray) -> int:
        """
        [FIX S1] Dùng np.random.choice thay vì stochastic_accept vòng lặp vô tận.
        O(n) và đảm bảo luôn trả về kết quả.
        """
        # Đảm bảo probs hợp lệ
        probs = np.clip(probs, 0, None)
        total = probs.sum()
        if total <= 0:
            return random.choice(candidates)
        probs = probs / total
        chosen_idx = np.random.choice(len(candidates), p=probs)
        return candidates[chosen_idx]

    @staticmethod
    def _count_valid_vehicles(path: list) -> int:
        """
        [FIX S4] Đếm chỉ những route thực sự có khách hàng (không đếm route rỗng).
        Route rỗng = 0 → 0 liên tiếp (không phục vụ ai).
        """
        count = 0
        prev = path[0]
        for node in path[1:]:
            if node == 0 and prev != 0:
                count += 1
            prev = node
        return count