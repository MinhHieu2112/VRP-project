import numpy as np
from cvrp_base import CVRPGraph


class Ant:
    def __init__(self, graph: CVRPGraph, start_index=0):
        self.graph = graph
        self.current_index = start_index
        self.vehicle_load = 0
        self.travel_path = [start_index]

        # [FIX P1] Dùng set thay vì list để remove O(1) thay vì O(n)
        self._index_to_visit_set = set(range(graph.node_num))
        self._index_to_visit_set.discard(start_index)

        self.total_travel_distance = 0.0

    @property
    def index_to_visit(self):
        """Trả về sorted list để numpy fancy indexing ổn định."""
        return sorted(self._index_to_visit_set)

    def index_to_visit_empty(self):
        return len(self._index_to_visit_set) == 0

    def move_to_next_index(self, next_index):
        self.travel_path.append(next_index)
        self.total_travel_distance += self.graph.node_dist_mat[self.current_index][next_index]

        if self.graph.nodes[next_index].is_depot:
            self.vehicle_load = 0
        else:
            self.vehicle_load += self.graph.nodes[next_index].demand
            # [FIX P1] O(1) remove với set
            self._index_to_visit_set.discard(next_index)

        self.current_index = next_index

    def check_condition(self, next_index) -> bool:
        """Kiểm tra capacity constraint."""
        if next_index == 0:
            return True
        return (self.vehicle_load + self.graph.nodes[next_index].demand
                <= self.graph.vehicle_capacity)

    def cal_next_index_meet_constrains(self):
        """Trả về các node khả thi (thỏa capacity)."""
        return [i for i in self.index_to_visit if self.check_condition(i)]

    # [FIX S4] Kiểm tra route rỗng: không cho phép depot→depot liên tiếp
    def is_at_depot(self):
        return self.current_index == 0

    @staticmethod
    def cal_total_travel_distance(graph: CVRPGraph, travel_path):
        distance = 0.0
        current_ind = travel_path[0]
        for next_ind in travel_path[1:]:
            distance += graph.node_dist_mat[current_ind][next_ind]
            current_ind = next_ind
        return distance