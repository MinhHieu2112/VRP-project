import numpy as np
import copy


class Node:
    def __init__(self, id: int, x: float, y: float, demand: float):
        super()
        self.id = id
        self.is_depot = (id == 0)
        self.x = x
        self.y = y
        self.demand = demand


class CVRPGraph:
    def __init__(self, node_num, nodes, node_dist_mat, vehicle_capacity,
                 rho=0.1,          # Global evaporation rate
                 xi=0.01):         # [FIX C4] Local update rate riêng biệt (ACS gốc dùng ξ ≈ 0.01–0.1)
        super()
        self.node_num = node_num
        self.nodes = nodes
        self.node_dist_mat = node_dist_mat.astype(np.float64)
        self.vehicle_capacity = vehicle_capacity
        self.rho = rho
        self.xi = xi   # [FIX C4] tách biệt local update rate

        # Khởi tạo pheromone từ NNH solution
        # [FIX S2] Không giới hạn max_vehicle_num để đảm bảo solution đầy đủ
        self.nnh_travel_path, nnh_distance, _ = self.nearest_neighbor_heuristic()
        if nnh_distance <= 0:
            nnh_distance = 1.0  # guard tránh chia 0
        self.init_pheromone_val = 1.0 / (nnh_distance * self.node_num)

        self.pheromone_mat = np.full(
            (self.node_num, self.node_num),
            self.init_pheromone_val,
            dtype=np.float64
        )

        # [FIX S3] Heuristic matrix: xử lý đúng
        # Chỉ zero diagonal, KHÔNG gán 1e-10 cho off-diagonal zeros
        heuristic_mat = self.node_dist_mat.copy()
        # Off-diagonal zeros = 2 địa điểm trùng vị trí → heuristic = 0 (không hấp dẫn đặc biệt)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.heuristic_info_mat = np.where(heuristic_mat > 0, 1.0 / heuristic_mat, 0.0)
        # Diagonal luôn = 0 (không có self-loop)
        np.fill_diagonal(self.heuristic_info_mat, 0.0)

    # ------------------------------------------------------------------ #
    #  Pheromone Updates                                                   #
    # ------------------------------------------------------------------ #

    def local_update_pheromone(self, start_ind, end_ind):
        """
        [FIX C4] Local update dùng xi (ξ) riêng, không dùng chung rho.
        Công thức ACS: τ_ij ← (1-ξ)·τ_ij + ξ·τ_0
        """
        self.pheromone_mat[start_ind][end_ind] = (
            (1 - self.xi) * self.pheromone_mat[start_ind][end_ind]
            + self.xi * self.init_pheromone_val
        )

    def global_update_pheromone(self, best_path, best_path_distance):
        """
        Global update chỉ reinforce best path, evaporate toàn bộ.
        Công thức ACS: τ_ij ← (1-ρ)·τ_ij + ρ/L_best
        """
        if best_path_distance <= 0:
            return
        self.pheromone_mat *= (1 - self.rho)
        current_ind = best_path[0]
        for next_ind in best_path[1:]:
            self.pheromone_mat[current_ind][next_ind] += self.rho / best_path_distance
            current_ind = next_ind

    # ------------------------------------------------------------------ #
    #  Nearest Neighbor Heuristic (để khởi tạo pheromone)                 #
    # ------------------------------------------------------------------ #

    def nearest_neighbor_heuristic(self):
        """
        [FIX S2] Bỏ giới hạn max_vehicle_num.
        NNH phải cover TẤT CẢ customers để init_pheromone_val hợp lệ.
        """
        index_to_visit = list(range(1, self.node_num))
        current_index = 0
        current_load = 0
        travel_distance = 0.0
        travel_path = [0]

        while index_to_visit:
            nearest = self._cal_nearest_next_index(index_to_visit, current_index, current_load)
            if nearest is None:
                # Quay về depot, bắt đầu vehicle mới
                travel_distance += self.node_dist_mat[current_index][0]
                travel_path.append(0)
                current_index = 0
                current_load = 0
            else:
                current_load += self.nodes[nearest].demand
                travel_distance += self.node_dist_mat[current_index][nearest]
                travel_path.append(nearest)
                current_index = nearest
                index_to_visit.remove(nearest)

        travel_distance += self.node_dist_mat[current_index][0]
        travel_path.append(0)
        vehicle_num = travel_path.count(0) - 1
        return travel_path, travel_distance, vehicle_num

    def _cal_nearest_next_index(self, index_to_visit, current_index, current_load):
        nearest_ind = None
        nearest_distance = None
        for next_index in index_to_visit:
            if current_load + self.nodes[next_index].demand > self.vehicle_capacity:
                continue
            dist = self.node_dist_mat[current_index][next_index]
            if nearest_distance is None or dist < nearest_distance:
                nearest_distance = dist
                nearest_ind = next_index
        return nearest_ind

    @staticmethod
    def calculate_dist(node_a, node_b):
        return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))


class PathMessage:
    def __init__(self, path, distance):
        if path is not None:
            self.path = copy.deepcopy(path)
            self.distance = copy.deepcopy(distance)
            self.used_vehicle_num = self.path.count(0) - 1
        else:
            self.path = None
            self.distance = None
            self.used_vehicle_num = None

    def get_path_info(self):
        return self.path, self.distance, self.used_vehicle_num