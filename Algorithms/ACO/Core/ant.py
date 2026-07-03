# File định nghĩa lớp Ant mô phỏng hành vi của một con kiến trong thuật toán ACO.
import numpy as np
from Models.cvrp_base import CVRPGraph

class Ant:
    """Đại diện cho một con kiến thực hiện chuyến đi trong bài toán VRP."""

    def __init__(self, graph: CVRPGraph, start_index: int = 0):
        # Khởi tạo con kiến tại depot với danh sách node cần thăm.
        self.graph         = graph
        self.current_index = start_index
        self.vehicle_load  = 0
        self.travel_path   = [start_index]

        self._index_to_visit_set = set(range(graph.node_num))
        self._index_to_visit_set.discard(start_index)

        self.total_travel_distance = 0.0

    @property
    def index_to_visit(self) -> list:
        # Trả về danh sách các node chưa được thăm theo thứ tự tăng dần.
        return sorted(self._index_to_visit_set)

    def index_to_visit_empty(self) -> bool:
        # Kiểm tra xem toàn bộ khách hàng đã được thăm hay chưa.
        return len(self._index_to_visit_set) == 0

    def move_to_next_index(self, next_index: int):
        # Di chuyển con kiến đến node tiếp theo và cập nhật trạng thái.
        self.total_travel_distance += (
            self.graph.node_dist_mat[self.current_index][next_index]
        )
        self.travel_path.append(next_index)

        if self.graph.nodes[next_index].is_depot:
            self.vehicle_load = 0
        else:
            self.vehicle_load += self.graph.nodes[next_index].demand
            self._index_to_visit_set.discard(next_index)

        self.current_index = next_index

    def check_condition(self, next_index: int) -> bool:
        # Kiểm tra ràng buộc tải trọng trước khi di chuyển đến node tiếp theo.
        if next_index == 0:
            return True
        return (self.vehicle_load + self.graph.nodes[next_index].demand
                <= self.graph.vehicle_capacity)

    def cal_next_index_meet_constrains(self) -> list:
        # Tính danh sách các node khả thi thỏa mãn ràng buộc tải trọng.
        return [i for i in self.index_to_visit if self.check_condition(i)]

    def is_at_depot(self) -> bool:
        # Kiểm tra xem con kiến đang đứng tại kho depot hay không.
        return self.current_index == 0

    def force_visit_remaining(self, remaining_nodes: list):
        # Cưỡng bức thăm tất cả các node còn lại dù vi phạm ràng buộc tải trọng.
        for node in remaining_nodes:
            if not self.check_condition(node):
                if not self.is_at_depot():
                    self.move_to_next_index(0)
            self.move_to_next_index(node)

    @staticmethod
    def cal_total_travel_distance(graph: CVRPGraph, travel_path: list) -> float:
        # Tính lại tổng khoảng cách của một đường đi nhằm mục đích kiểm tra.
        distance    = 0.0
        current_ind = travel_path[0]
        for next_ind in travel_path[1:]:
            distance   += graph.node_dist_mat[current_ind][next_ind]
            current_ind = next_ind
        return distance