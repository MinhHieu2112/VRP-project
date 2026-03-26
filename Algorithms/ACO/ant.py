import numpy as np
import copy
from cvrp_base import CVRPGraph
from threading import Event


class Ant:
    def __init__(self, graph: CVRPGraph, start_index=0):
        super()
        self.graph = graph
        self.current_index = start_index
        self.vehicle_load = 0
        self.travel_path = [start_index]
        self.index_to_visit = list(range(graph.node_num))
        self.index_to_visit.remove(start_index)
        self.total_travel_distance = 0

    def clear(self):
        self.travel_path.clear()
        self.index_to_visit.clear()

    def move_to_next_index(self, next_index):
        self.travel_path.append(next_index)
        self.total_travel_distance += self.graph.node_dist_mat[self.current_index][next_index]

        if self.graph.nodes[next_index].is_depot:
            self.vehicle_load = 0
        else:
            self.vehicle_load += self.graph.nodes[next_index].demand

        self.current_index = next_index
        if next_index in self.index_to_visit:
            self.index_to_visit.remove(next_index)

    def index_to_visit_empty(self):
        return len(self.index_to_visit) == 0

    def check_condition(self, next_index) -> bool:
        if next_index == 0:
            return True
        if self.vehicle_load + self.graph.nodes[next_index].demand > self.graph.vehicle_capacity:
            return False
        return True

    def cal_next_index_meet_constrains(self):
        next_index_meet_constrains = []
        for next_index in self.index_to_visit:
            if self.check_condition(next_index):
                next_index_meet_constrains.append(next_index)
        return next_index_meet_constrains

    @staticmethod
    def cal_total_travel_distance(graph: CVRPGraph, travel_path):
        distance = 0
        current_ind = travel_path[0]
        for next_ind in travel_path[1:]:
            distance += graph.node_dist_mat[current_ind][next_ind]
            current_ind = next_ind
        return distance