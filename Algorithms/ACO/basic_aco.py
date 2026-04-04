import numpy as np
import random
import time
from cvrp_base import CVRPGraph
from ant import Ant


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
                    # FIX #6: Đếm vehicle đúng bằng hàm đã sửa
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
        max_steps = self.graph.node_num * 3
        steps = 0

        while not ant.index_to_visit_empty():
            steps += 1
            if steps > max_steps:
                for remaining in ant.index_to_visit:
                    if ant.current_index != 0:
                        ant.move_to_next_index(0)
                        self.graph.local_update_pheromone(ant.current_index, 0)
                    ant.move_to_next_index(remaining)
                    self.graph.local_update_pheromone(ant.current_index, remaining)
                break

            feasible = ant.cal_next_index_meet_constrains()

            if not feasible:
                if not ant.is_at_depot():
                    ant.move_to_next_index(0)
                    self.graph.local_update_pheromone(ant.current_index, 0)
                else:
                    print('[WARNING] Ant bị kẹt tại depot.')
                    break
            else:
                next_index = self.select_next_index(ant, feasible)
                ant.move_to_next_index(next_index)
                self.graph.local_update_pheromone(ant.current_index, next_index)

        if not ant.is_at_depot():
            ant.move_to_next_index(0)
            self.graph.local_update_pheromone(ant.current_index, 0)

    def select_next_index(self, ant: Ant, feasible_nodes: list) -> int:
        current_index = ant.current_index

        pheromone = self.graph.pheromone_mat[current_index][feasible_nodes]
        heuristic = self.graph.heuristic_info_mat[current_index][feasible_nodes]

        scores = (np.power(pheromone, self.alpha)
                  * np.power(heuristic, self.beta))

        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        score_sum = scores.sum()
        if score_sum <= 0:
            return random.choice(feasible_nodes)

        if random.random() < self.q0:
            best_local_idx = int(np.argmax(scores))
            return feasible_nodes[best_local_idx]
        else:
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
        FIX #6: Đếm số route thực sự có ít nhất 1 customer.
        
        Phiên bản cũ: đếm số lần gặp node=0 với prev!=0.
        Lỗi: path=[0,0,1,2,0] → count=1 dù có route rỗng [0,0] ở đầu.
        
        Phiên bản mới: tách path thành từng route, chỉ đếm route có customer.
        """
        count = 0
        has_customer = False  # Cờ: route hiện tại có customer không

        for node in path[1:]:  # Bỏ depot đầu tiên
            if node == 0:
                if has_customer:
                    count += 1
                has_customer = False  # Reset cho route mới
            else:
                has_customer = True

        # Route cuối chưa kết thúc bằng 0 (edge case)
        if has_customer:
            count += 1

        return count