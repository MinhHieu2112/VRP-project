"""
Algorithms/ACO/Core/ant.py
===========================
Lớp Ant — đại diện một kiến trong ACS/ACO cho ACVRP.

Kiến xây lộ trình bằng cách:
  1. Bắt đầu tại depot (node 0)
  2. Chọn node tiếp theo theo ACS transition rule (trong engine.py)
  3. Di chuyển, cập nhật tải và chi phí
  4. Khi vượt capacity hoặc hết node → về depot (mở tuyến mới)
  5. Kết thúc tại depot

Không có thay đổi logic lớn so với phiên bản cũ — các fix chính ở engine.py.
Cải thiện: comment rõ hơn về tính bất đối xứng của ACVRP.
"""

import numpy as np
from Models.cvrp_base import CVRPGraph


class Ant:
    def __init__(self, graph: CVRPGraph, start_index: int = 0):
        self.graph         = graph
        self.current_index = start_index
        self.vehicle_load  = 0
        self.travel_path   = [start_index]

        # Set để remove O(1) — nhanh hơn list với bài toán lớn
        self._index_to_visit_set = set(range(graph.node_num))
        self._index_to_visit_set.discard(start_index)  # loại depot

        self.total_travel_distance = 0.0

    @property
    def index_to_visit(self) -> list:
        """Trả về sorted list để numpy fancy indexing ổn định."""
        return sorted(self._index_to_visit_set)

    def index_to_visit_empty(self) -> bool:
        return len(self._index_to_visit_set) == 0

    def move_to_next_index(self, next_index: int):
        """
        Di chuyển kiến đến next_index.

        Chi phí được tính theo d(current → next) — bất đối xứng ACVRP:
          d(i,j) ≠ d(j,i), dùng đúng chiều từ current đến next.
        """
        self.total_travel_distance += (
            self.graph.node_dist_mat[self.current_index][next_index]
        )
        self.travel_path.append(next_index)

        if self.graph.nodes[next_index].is_depot:
            # Về depot: reset tải xe, KHÔNG xóa depot khỏi to_visit
            self.vehicle_load = 0
        else:
            # Đến customer: tăng tải và đánh dấu đã thăm
            self.vehicle_load += self.graph.nodes[next_index].demand
            self._index_to_visit_set.discard(next_index)

        self.current_index = next_index

    def check_condition(self, next_index: int) -> bool:
        """
        Kiểm tra capacity constraint trước khi di chuyển đến next_index.
        Depot luôn hợp lệ (về depot để mở tuyến mới).
        """
        if next_index == 0:
            return True
        return (self.vehicle_load + self.graph.nodes[next_index].demand
                <= self.graph.vehicle_capacity)

    def cal_next_index_meet_constrains(self) -> list:
        """Trả về danh sách các node khả thi (thỏa capacity constraint)."""
        return [i for i in self.index_to_visit if self.check_condition(i)]

    def is_at_depot(self) -> bool:
        return self.current_index == 0

    def force_visit_remaining(self, remaining_nodes: list):
        """
        Force-visit các node còn lại khi vượt max_steps.

        Giữ capacity constraint:
        - Nếu node tiếp theo không vừa capacity → về depot trước
        - Sau đó mới đến node đó
        Đảm bảo mọi khách hàng đều được phục vụ (không bỏ sót).
        """
        for node in remaining_nodes:
            if not self.check_condition(node):
                # Không đủ tải → về depot reset
                if not self.is_at_depot():
                    self.move_to_next_index(0)
            self.move_to_next_index(node)

    @staticmethod
    def cal_total_travel_distance(graph: CVRPGraph, travel_path: list) -> float:
        """Tính lại tổng khoảng cách cho một path (dùng để verify)."""
        distance    = 0.0
        current_ind = travel_path[0]
        for next_ind in travel_path[1:]:
            distance   += graph.node_dist_mat[current_ind][next_ind]
            current_ind = next_ind
        return distance