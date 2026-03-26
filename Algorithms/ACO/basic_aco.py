import numpy as np
import random
from cvrp_base import CVRPGraph, PathMessage
from ant import Ant
from threading import Thread
from queue import Queue
import time


class BasicACO:
    def __init__(self, graph: CVRPGraph, ants_num=10, max_iter=200, beta=2, q0=0.1,
                 whether_or_not_to_show_figure=True):
        super()
        self.graph = graph
        self.ants_num = ants_num
        self.max_iter = max_iter
        self.max_load = graph.vehicle_capacity
        self.beta = beta
        self.q0 = q0
        self.best_path_distance = None
        self.best_path = None
        self.best_vehicle_num = None
        self.whether_or_not_to_show_figure = whether_or_not_to_show_figure

    def run_basic_aco(self):
        path_queue_for_figure = Queue()
        basic_aco_thread = Thread(target=self._basic_aco, args=(path_queue_for_figure,))
        basic_aco_thread.start()
        basic_aco_thread.join()
        return self.best_path, self.best_path_distance, self.best_vehicle_num

    def _basic_aco(self, path_queue_for_figure: Queue):
        start_time_total = time.time()
        start_iteration = 0
        for iter in range(self.max_iter):
            ants = [Ant(self.graph) for _ in range(self.ants_num)]
            for k in range(self.ants_num):
                while not ants[k].index_to_visit_empty():
                    next_index = self.select_next_index(ants[k])
                    if not ants[k].check_condition(next_index):
                        next_index = 0
                    ants[k].move_to_next_index(next_index)
                    self.graph.local_update_pheromone(ants[k].current_index, next_index)
                ants[k].move_to_next_index(0)
                self.graph.local_update_pheromone(ants[k].current_index, 0)

            paths_distance = np.array([ant.total_travel_distance for ant in ants])
            best_index = np.argmin(paths_distance)
            if self.best_path is None or paths_distance[best_index] < self.best_path_distance:
                self.best_path = ants[best_index].travel_path
                self.best_path_distance = paths_distance[best_index]
                self.best_vehicle_num = self.best_path.count(0) - 1
                start_iteration = iter
                print(f'[iteration {iter}]: find improved path, distance {self.best_path_distance}')

            self.graph.global_update_pheromone(self.best_path, self.best_path_distance)

            if iter - start_iteration > 50:
                print('No improvement in 50 iterations, stopping')
                break

        print(f'Final best path distance: {self.best_path_distance}, vehicles: {self.best_vehicle_num}')
        print(f'Time: {time.time() - start_time_total:.3f}s')

    def select_next_index(self, ant):
        current_index = ant.current_index
        index_to_visit = ant.index_to_visit

        if not index_to_visit:
            return 0

        transition_prob = self.graph.pheromone_mat[current_index][index_to_visit] * \
            np.power(self.graph.heuristic_info_mat[current_index][index_to_visit], self.beta)

        # Handle any nan or inf values
        transition_prob = np.nan_to_num(transition_prob, nan=0.0, posinf=1e10, neginf=0.0)

        prob_sum = np.sum(transition_prob)
        if prob_sum <= 0 or np.isnan(prob_sum) or np.isinf(prob_sum):
            # If all transition probabilities are zero/negative/nan/inf, fall back to random selection
            next_index = np.random.choice(index_to_visit)
        else:
            transition_prob = transition_prob / prob_sum

            if np.random.rand() < self.q0:
                max_prob_index = np.argmax(transition_prob)
                next_index = index_to_visit[max_prob_index]
            else:
                next_index = self.stochastic_accept(index_to_visit, transition_prob)
        return next_index

    @staticmethod
    def stochastic_accept(index_to_visit, transition_prob):
        N = len(index_to_visit)
        # transition_prob should already be normalized, but ensure it sums to 1
        sum_tran_prob = np.sum(transition_prob)
        if sum_tran_prob > 0:
            norm_transition_prob = transition_prob / sum_tran_prob
        else:
            # Fallback to uniform distribution
            norm_transition_prob = np.ones(N) / N

        while True:
            ind = int(N * random.random())
            if random.random() <= norm_transition_prob[ind]:
                return index_to_visit[ind]