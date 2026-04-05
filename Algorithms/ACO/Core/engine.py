import numpy as np
import random
import time
from Models.cvrp_base import CVRPGraph
from Core.ant import Ant


class BasicACO:
    def __init__(self, graph: CVRPGraph,
                 ants_num=10,
                 max_iter=200,
                 alpha=1,
                 beta=2,
                 q0=0.9,
                 no_improve_limit=50):
        self.graph = graph
        self.ants_num = ants_num
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.no_improve_limit = no_improve_limit

        self.best_path_distance = None
        self.best_path = None
        self.best_vehicle_num = None

    def run_basic_aco(self):
        self._basic_aco()
        return self.best_path, self.best_path_distance, self.best_vehicle_num

    def _basic_aco(self):
        start_time = time.time()
        no_improve_count = 0

        for iteration in range(self.max_iter):
            ants = [Ant(self.graph) for _ in range(self.ants_num)]

            for ant in ants:
                self._construct_solution(ant)

            improved = False
            for ant in ants:
                if (self.best_path is None
                        or ant.total_travel_distance < self.best_path_distance):
                    self.best_path = ant.travel_path[:]
                    self.best_path_distance = ant.total_travel_distance
                    self.best_vehicle_num = self._count_valid_vehicles(self.best_path)
                    improved = True
                    print(f'[iter {iteration}] Improved: '
                          f'dist={self.best_path_distance:.2f}, '
                          f'vehicles={self.best_vehicle_num}')

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1

            self.graph.global_update_pheromone(self.best_path, self.best_path_distance)

            if no_improve_count >= self.no_improve_limit:
                print(f'Early stop tại iteration {iteration}')
                break

        print(f'[DONE] dist={self.best_path_distance:.2f}, '
              f'vehicles={self.best_vehicle_num}, '
              f'time={time.time()-start_time:.2f}s')

    def _construct_solution(self, ant: Ant):
        """
        [FIX S1] max_steps được tính đủ lớn để cover mọi trường hợp:
          - N-1 bước thăm customer
          - Tối đa N-1 lần về depot (worst case: mỗi xe chở 1 đơn)
          - Buffer 10 bước phòng ngừa
        Tổng: 2*(N-1) + 10, thay vì N*3 cũ (vẫn đủ nhưng rõ ràng hơn về ý nghĩa).

        [FIX S2] Fallback khi vượt max_steps dùng ant.force_visit_remaining()
        để đảm bảo capacity constraint vẫn được giữ (không bỏ node nào).
        """
        n_customers = self.graph.node_num - 1  # trừ depot
        max_steps = 2 * n_customers + 10

        steps = 0
        while not ant.index_to_visit_empty():
            steps += 1

            # [FIX S2] Fallback an toàn: dùng force_visit_remaining giữ capacity
            if steps > max_steps:
                remaining = ant.index_to_visit  # sorted list
                print(f'[WARN] max_steps vượt quá ({max_steps}), '
                      f'force-visit {len(remaining)} node còn lại (capacity được giữ)')
                ant.force_visit_remaining(remaining)
                # Cập nhật pheromone cho các cạnh forced (dùng init_pheromone_val)
                # Không gọi local_update để tránh reinforce đường xấu
                break

            feasible = ant.cal_next_index_meet_constrains()

            if not feasible:
                # Không có node khả thi → về depot để reset load
                if not ant.is_at_depot():
                    prev = ant.current_index
                    ant.move_to_next_index(0)
                    self.graph.local_update_pheromone(prev, 0)
                else:
                    # [FIX S3] Ant kẹt tại depot: không thể xảy ra nếu validation
                    # đã đảm bảo demand ≤ capacity. Log và thoát an toàn.
                    print('[ERROR] Ant kẹt tại depot - kiểm tra lại demand/capacity. '
                          f'Nodes còn lại: {ant.index_to_visit}')
                    # Force visit với capacity check để không bỏ node
                    ant.force_visit_remaining(ant.index_to_visit)
                    break
            else:
                prev = ant.current_index
                next_index = self.select_next_index(ant, feasible)
                ant.move_to_next_index(next_index)
                self.graph.local_update_pheromone(prev, next_index)

        # Đảm bảo kết thúc tại depot
        if not ant.is_at_depot():
            prev = ant.current_index
            ant.move_to_next_index(0)
            self.graph.local_update_pheromone(prev, 0)

    def select_next_index(self, ant: Ant, feasible_nodes: list) -> int:
        """
        [FIX C1] Tối ưu tính toán score:
          - Nếu alpha == 1: bỏ np.power(pheromone, 1) → nhân trực tiếp
          - Nếu alpha != 1: dùng np.power như cũ
        Giảm overhead khi alpha=1 (trường hợp mặc định phổ biến nhất).

        feasible_nodes là list node ID (không phải index của scores array).
        np.argmax(scores) → index trong scores → feasible_nodes[idx] → node ID thật.
        Logic này đúng và được giữ nguyên.
        """
        current_index = ant.current_index

        # Fancy indexing: lấy pheromone và heuristic của các node khả thi
        pheromone = self.graph.pheromone_mat[current_index][feasible_nodes]
        heuristic = self.graph.heuristic_info_mat[current_index][feasible_nodes]

        # [FIX C1] Bỏ np.power khi alpha=1 để tiết kiệm tính toán
        if self.alpha == 1:
            scores = pheromone * np.power(heuristic, self.beta)
        else:
            scores = np.power(pheromone, self.alpha) * np.power(heuristic, self.beta)

        # Làm sạch NaN/Inf phòng ngừa
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        score_sum = scores.sum()
        if score_sum <= 0:
            # Tất cả scores = 0 (heuristic = 0, tức các node trùng vị trí)
            # → chọn ngẫu nhiên
            return random.choice(feasible_nodes)

        if random.random() < self.q0:
            # Exploitation: chọn node tốt nhất
            best_local_idx = int(np.argmax(scores))
            return feasible_nodes[best_local_idx]
        else:
            # Exploration: roulette wheel
            probs = scores / score_sum
            return self._roulette_wheel(feasible_nodes, probs)

    @staticmethod
    def _roulette_wheel(candidates: list, probs: np.ndarray) -> int:
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
        Đếm số route thực sự có ít nhất 1 customer.
        Tránh đếm route rỗng depot→depot.
        """
        count = 0
        has_customer = False

        for node in path[1:]:  # Bỏ depot đầu tiên
            if node == 0:
                if has_customer:
                    count += 1
                has_customer = False
            else:
                has_customer = True

        # Route cuối chưa kết thúc bằng 0 (edge case)
        if has_customer:
            count += 1

        return count