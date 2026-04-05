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
                 rho=0.1,    # Global evaporation rate
                 xi=0.01):   # Local update rate (ACS: ξ ≈ 0.01–0.1)
        super()
        self.node_num = node_num
        self.nodes = nodes
        self.node_dist_mat = node_dist_mat.astype(np.float64)
        self.vehicle_capacity = vehicle_capacity
        self.rho = rho
        self.xi = xi

        # [FIX V1] Validate dữ liệu đầu vào trước khi chạy ACO
        self._validate_inputs()

        # Khởi tạo pheromone từ NNH solution
        self.nnh_travel_path, nnh_distance, _ = self.nearest_neighbor_heuristic()
        if nnh_distance <= 0:
            nnh_distance = 1.0  # guard tránh chia 0
        self.init_pheromone_val = 1.0 / (nnh_distance * self.node_num)

        self.pheromone_mat = np.full(
            (self.node_num, self.node_num),
            self.init_pheromone_val,
            dtype=np.float64
        )

        # Heuristic matrix: 1/distance, diagonal = 0, zero off-diag = 0
        heuristic_mat = self.node_dist_mat.copy()
        with np.errstate(divide='ignore', invalid='ignore'):
            self.heuristic_info_mat = np.where(heuristic_mat > 0, 1.0 / heuristic_mat, 0.0)
        np.fill_diagonal(self.heuristic_info_mat, 0.0)

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def _validate_inputs(self):
        """
        [FIX V1] Kiểm tra tính hợp lệ của dữ liệu đầu vào.
        Phát hiện sớm các lỗi khiến ant bị kẹt vô hạn hoặc lời giải không hợp lệ.
        """
        errors = []
        warnings = []

        # 1. Ma trận khoảng cách phải vuông và khớp với node_num
        mat = self.node_dist_mat
        if mat.shape[0] != mat.shape[1]:
            errors.append(f"Ma trận khoảng cách không vuông: {mat.shape}")
        if mat.shape[0] != self.node_num:
            errors.append(
                f"Kích thước ma trận ({mat.shape[0]}) "
                f"không khớp node_num ({self.node_num})"
            )

        # 2. Kiểm tra NaN, Inf, và giá trị âm
        if np.any(np.isnan(mat)):
            errors.append("Ma trận khoảng cách chứa NaN")
        if np.any(np.isinf(mat)):
            errors.append("Ma trận khoảng cách chứa Inf")

        # Giá trị âm: phân biệt floating-point rounding error vs lỗi thật sự.
        # OSRM đôi khi trả về giá trị âm rất nhỏ (vd: -0.0007) do làm tròn số thực.
        # Ngưỡng tolerance = -0.01 km: nhỏ hơn → clip về 0 (warning).
        #                              lớn hơn → lỗi thật sự (error).
        NEG_TOLERANCE = -0.01
        neg_mask = mat < 0
        if np.any(neg_mask):
            min_neg = mat[neg_mask].min()
            neg_count = neg_mask.sum()
            if min_neg >= NEG_TOLERANCE:
                # Floating-point noise → clip về 0 và tiếp tục
                warnings.append(
                    f"Ma trận có {neg_count} giá trị âm nhỏ "
                    f"(min={min_neg:.6f}, ngưỡng={NEG_TOLERANCE}). "
                    f"Đây là floating-point rounding error từ OSRM → tự động clip về 0."
                )
                self.node_dist_mat = np.clip(self.node_dist_mat, 0.0, None)
            else:
                # Giá trị âm lớn → lỗi thật sự trong dữ liệu
                errors.append(
                    f"Ma trận chứa {neg_count} giá trị âm thật sự "
                    f"(min={min_neg:.6f}). Kiểm tra lại nguồn dữ liệu."
                )

        # 3. Đường chéo phải = 0
        diag_nonzero = np.sum(np.diag(mat) != 0)
        if diag_nonzero > 0:
            warnings.append(f"Đường chéo có {diag_nonzero} giá trị ≠ 0, sẽ bị bỏ qua")

        # 4. [FIX V1 CORE] Mỗi customer phải có demand ≤ vehicle_capacity
        # Nếu vi phạm, ant sẽ bị kẹt vĩnh viễn tại depot
        infeasible = []
        for node in self.nodes:
            if not node.is_depot and node.demand > self.vehicle_capacity:
                infeasible.append((node.id, node.demand))
        if infeasible:
            errors.append(
                f"Có {len(infeasible)} node có demand > vehicle_capacity "
                f"({self.vehicle_capacity}): {infeasible[:5]}"
                + (" ..." if len(infeasible) > 5 else "")
            )

        # 5. Phải có ít nhất 1 customer (ngoài depot)
        if self.node_num < 2:
            errors.append("Cần ít nhất 1 customer ngoài depot")

        # In warnings
        for w in warnings:
            print(f"[WARN] {w}")

        # Dừng nếu có lỗi nghiêm trọng
        if errors:
            msg = "\n".join(f"  - {e}" for e in errors)
            raise ValueError(f"Dữ liệu đầu vào không hợp lệ:\n{msg}")

        print(f"[OK] Validation passed: {self.node_num} nodes, "
              f"capacity={self.vehicle_capacity}")

    # ------------------------------------------------------------------ #
    #  Pheromone Updates                                                   #
    # ------------------------------------------------------------------ #

    def local_update_pheromone(self, start_ind, end_ind):
        """
        Local update dùng xi (ξ) riêng, không dùng chung rho.
        Công thức ACS: τ_ij ← (1-ξ)·τ_ij + ξ·τ_0
        Cập nhật cả 2 chiều vì pheromone matrix là symmetric trong khởi tạo.
        """
        new_val = ((1 - self.xi) * self.pheromone_mat[start_ind][end_ind]
                   + self.xi * self.init_pheromone_val)
        self.pheromone_mat[start_ind][end_ind] = new_val
        # [FIX P2] Cập nhật chiều ngược để giữ tính nhất quán với init symmetric
        self.pheromone_mat[end_ind][start_ind] = new_val

    def global_update_pheromone(self, best_path, best_path_distance):
        """
        [FIX P2] Global update cập nhật đủ 2 chiều i→j và j→i.

        Lý do: Ma trận pheromone được khởi tạo symmetric (τ_ij = τ_ji = τ_0).
        Local update đã giữ symmetry. Nếu global update chỉ cập nhật 1 chiều,
        τ_ij ≠ τ_ji → mâu thuẫn với local update, gây bias không mong muốn.

        Với CVRP thực tế (đường 2 chiều bằng nhau), symmetric là hợp lý.
        Nếu bài toán là asymmetric VRP, chỉ cập nhật 1 chiều và bỏ comment này.
        """
        if best_path_distance <= 0:
            return

        # Evaporate toàn bộ
        self.pheromone_mat *= (1 - self.rho)

        # Reinforce best path: cả 2 chiều
        delta = self.rho / best_path_distance
        current_ind = best_path[0]
        for next_ind in best_path[1:]:
            self.pheromone_mat[current_ind][next_ind] += delta
            # [FIX P2] Chiều ngược
            self.pheromone_mat[next_ind][current_ind] += delta
            current_ind = next_ind

    # ------------------------------------------------------------------ #
    #  Nearest Neighbor Heuristic (để khởi tạo pheromone)                 #
    # ------------------------------------------------------------------ #

    def nearest_neighbor_heuristic(self):
        """
        NNH phải cover TẤT CẢ customers để init_pheromone_val hợp lệ.
        Không giới hạn max_vehicle_num.
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