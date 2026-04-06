import numpy as np
import random
import time
from Models.cvrp_base import CVRPGraph
from Core.ant import Ant


class BasicACO:
    def __init__(self, graph: CVRPGraph,
                 ants_num=10,
                 max_iter=50000,
                 alpha=1,
                 beta=2,
                 q0=0.9,
                 no_improve_limit=5000):
        """
        Parameters
        ----------
        max_iter        : Safety cap — số vòng lặp tối đa tuyệt đối.
        no_improve_limit: Dừng sớm nếu sau n vòng liên tiếp không cải thiện best.
        """
        self.graph            = graph
        self.ants_num         = ants_num
        self.max_iter         = max_iter
        self.alpha            = alpha
        self.beta             = beta
        self.q0               = q0
        self.no_improve_limit = no_improve_limit

        self.best_path_distance = None
        self.best_path          = None
        self.best_vehicle_num   = None

    def run_basic_aco(self):
        self._basic_aco()
        return self.best_path, self.best_path_distance, self.best_vehicle_num

    def _basic_aco(self):
        start_time       = time.time()
        no_improve_count = 0

        for iteration in range(self.max_iter):

            # --- Điều kiện dừng no-improvement ---
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
                    self.best_vehicle_num   = self._count_valid_vehicles(self.best_path)
                    improved                = True
                    print(f"[iter {iteration:5d}] Improved: "
                          f"dist={self.best_path_distance:.0f}m "
                          f"({self.best_path_distance/1000:.2f}km), "
                          f"vehicles={self.best_vehicle_num}")

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1

            self.graph.global_update_pheromone(self.best_path, self.best_path_distance)

            if iteration % 50 == 0 and iteration > 0:
                elapsed = time.time() - start_time
                print(f"  [iter {iteration:5d}] elapsed={elapsed:.1f}s | "
                      f"NoImprove={no_improve_count}/{self.no_improve_limit}")

        elapsed = time.time() - start_time
        print(f"[DONE] dist={self.best_path_distance:.0f}m "
              f"({self.best_path_distance/1000:.2f}km), "
              f"vehicles={self.best_vehicle_num}, "
              f"time={elapsed:.2f}s")

    def _construct_solution(self, ant: Ant):
        n_customers = self.graph.node_num - 1
        max_steps   = 2 * n_customers + 10

        steps = 0
        while not ant.index_to_visit_empty():
            steps += 1

            if steps > max_steps:
                remaining = ant.index_to_visit
                print(f"[WARN] max_steps vượt quá ({max_steps}), "
                      f"force-visit {len(remaining)} node còn lại")
                ant.force_visit_remaining(remaining)
                break

            feasible = ant.cal_next_index_meet_constrains()

            if not feasible:
                if not ant.is_at_depot():
                    prev = ant.current_index
                    ant.move_to_next_index(0)
                    self.graph.local_update_pheromone(prev, 0)
                else:
                    print('[ERROR] Ant kẹt tại depot.')
                    ant.force_visit_remaining(ant.index_to_visit)
                    break
            else:
                prev       = ant.current_index
                next_index = self.select_next_index(ant, feasible)
                ant.move_to_next_index(next_index)
                self.graph.local_update_pheromone(prev, next_index)

        if not ant.is_at_depot():
            prev = ant.current_index
            ant.move_to_next_index(0)
            self.graph.local_update_pheromone(prev, 0)

    def select_next_index(self, ant: Ant, feasible_nodes: list) -> int:
        current_index = ant.current_index

        pheromone = self.graph.pheromone_mat[current_index][feasible_nodes]
        heuristic = self.graph.heuristic_info_mat[current_index][feasible_nodes]

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
            return feasible_nodes[best_local_idx]
        else:
            probs = scores / score_sum
            return self._roulette_wheel(feasible_nodes, probs)

    @staticmethod
    def _roulette_wheel(candidates: list, probs: np.ndarray) -> int:
        probs      = np.clip(probs, 0, None)
        total      = probs.sum()
        if total <= 0:
            return random.choice(candidates)
        probs      = probs / total
        chosen_idx = np.random.choice(len(candidates), p=probs)
        return candidates[chosen_idx]

    @staticmethod
    def _count_valid_vehicles(path: list) -> int:
        count        = 0
        has_customer = False

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