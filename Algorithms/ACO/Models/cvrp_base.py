"""
Algorithms/ACO/Models/cvrp_base.py  — FIX ACVRP
=================================================
[FIX-ACVRP-1] Pheromone matrix phải BẤT ĐỐI XỨNG cho ACVRP.

Lỗi cũ:
  - local_update_pheromone  cập nhật cả 2 chiều τ_ij = τ_ji (symmetric)
  - global_update_pheromone cập nhật cả 2 chiều τ_ij = τ_ji (symmetric)

Ràng buộc báo cáo (mục 5.2):
  d(i,j) ≠ d(j,i)  — ma trận OSRM là bất đối xứng.
  Pheromone phải phản ánh đúng chiều di chuyển: τ_ij ≠ τ_ji.
  Cập nhật 2 chiều bằng nhau làm mất tính asymmetric, kiến sẽ chọn
  đường theo phero symmetric trong khi chi phí thực là asymmetric
  → nghiệm sai.

Fix: chỉ cập nhật 1 chiều i→j cho cả local và global update.
"""

import numpy as np
import copy


class Node:
    def __init__(self, id: int, x: float, y: float, demand: float):
        """Khởi tạo node với id, toạ độ và demand."""
        super()
        self.id       = id
        self.is_depot = (id == 0)
        self.x        = x
        self.y        = y
        self.demand   = demand


class CVRPGraph:
    def __init__(self, node_num, nodes, node_dist_mat, vehicle_capacity,
                 rho=0.1,
                 xi=0.01):
        """
        Khởi tạo đồ thị ACVRP với pheromone, heuristic và validation.

        Parameters
        ----------
        rho : Tốc độ bay hơi global (0 < rho < 1).
        xi  : Tốc độ cập nhật local ACS (0 < xi < 1).
        """
        super()
        self.node_num         = node_num
        self.nodes            = nodes
        self.node_dist_mat    = node_dist_mat.astype(np.float64)
        self.vehicle_capacity = vehicle_capacity
        self.rho              = rho
        self.xi               = xi

        self._validate_inputs()

        # Khởi tạo pheromone từ NNH solution
        self.nnh_travel_path, nnh_distance, _ = self.nearest_neighbor_heuristic()
        if nnh_distance <= 0:
            nnh_distance = 1.0
        self.init_pheromone_val = 1.0 / (nnh_distance * self.node_num)

        # [FIX-ACVRP-1] Pheromone khởi tạo đều nhưng CẬP NHẬT 1 CHIỀU
        # Khởi tạo đều là hợp lý (không có thông tin ban đầu).
        # Quan trọng: sau khi chạy, τ_ij ≠ τ_ji nhờ update 1 chiều.
        self.pheromone_mat = np.full(
            (self.node_num, self.node_num),
            self.init_pheromone_val,
            dtype=np.float64
        )

        # Heuristic η_ij = 1/d(i,j) — bất đối xứng tự nhiên từ OSRM
        with np.errstate(divide='ignore', invalid='ignore'):
            self.heuristic_info_mat = np.where(
                self.node_dist_mat > 0,
                1.0 / self.node_dist_mat,
                0.0
            )
        np.fill_diagonal(self.heuristic_info_mat, 0.0)

    # ──────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────

    def _validate_inputs(self):
        """Kiểm tra tính hợp lệ dữ liệu đầu vào trước khi chạy ACO."""
        errors   = []
        warnings = []
        mat      = self.node_dist_mat

        if mat.shape[0] != mat.shape[1]:
            errors.append(f"Ma trận không vuông: {mat.shape}")
        if mat.shape[0] != self.node_num:
            errors.append(
                f"Kích thước ma trận ({mat.shape[0]}) "
                f"≠ node_num ({self.node_num})"
            )

        if np.any(np.isnan(mat)):
            errors.append("Ma trận chứa NaN")
        if np.any(np.isinf(mat)):
            errors.append("Ma trận chứa Inf")

        # Clip giá trị âm nhỏ (OSRM floating-point noise)
        NEG_TOLERANCE = -0.01
        neg_mask = mat < 0
        if np.any(neg_mask):
            min_neg   = mat[neg_mask].min()
            neg_count = int(neg_mask.sum())
            if min_neg >= NEG_TOLERANCE:
                warnings.append(
                    f"{neg_count} giá trị âm nhỏ (min={min_neg:.6f}) "
                    f"→ clip về 0 (OSRM rounding noise)."
                )
                self.node_dist_mat = np.clip(self.node_dist_mat, 0.0, None)
            else:
                errors.append(
                    f"{neg_count} giá trị âm lớn (min={min_neg:.4f}). "
                    f"Kiểm tra lại nguồn dữ liệu."
                )

        diag_nonzero = int(np.sum(np.diag(mat) != 0))
        if diag_nonzero > 0:
            warnings.append(f"Đường chéo có {diag_nonzero} giá trị ≠ 0")

        # Kiểm tra tính bất đối xứng (thông tin, không phải lỗi)
        diff = np.abs(mat - mat.T)
        asymmetric_count = int(np.sum(diff > 1.0))  # sai lệch >1m
        if asymmetric_count > 0:
            print(f"[INFO] Ma trận ACVRP bất đối xứng: "
                  f"{asymmetric_count} cặp (i,j) có d(i,j)≠d(j,i)")

        infeasible = [
            (node.id, node.demand)
            for node in self.nodes
            if not node.is_depot and node.demand > self.vehicle_capacity
        ]
        if infeasible:
            errors.append(
                f"{len(infeasible)} node có demand > capacity "
                f"({self.vehicle_capacity}): {infeasible[:5]}"
            )

        if self.node_num < 2:
            errors.append("Cần ít nhất 1 customer ngoài depot")

        for w in warnings:
            print(f"[WARN] {w}")

        if errors:
            raise ValueError(
                "Dữ liệu không hợp lệ:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        print(f"[OK] Validation passed: {self.node_num} nodes, "
              f"capacity={self.vehicle_capacity}, ACVRP asymmetric")

    # Trong CVRPGraph.__init__(), sau khi khởi tạo pheromone_mat:
    # Thêm optional seed
    def seed_pheromone(self, solution: list[list[int]], seed_weight: float = 2.0):
        """
        Reinforce pheromone dọc theo các cạnh của nghiệm seed.
        seed_weight: bội số so với init_pheromone_val (mặc định 2× = bias nhẹ).
        
        [ACVRP] Chỉ cập nhật 1 chiều i→j để giữ tính bất đối xứng.
        """
        boost = self.init_pheromone_val * seed_weight
        for route in solution:
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                # Chỉ boost nếu hiện tại đang ở init level, không ghi đè
                # nếu đã có global update chạy trước
                self.pheromone_mat[u][v] = max(
                    self.pheromone_mat[u][v],
                    boost
                )

    # ──────────────────────────────────────────────────────────────────
    # Pheromone Updates — BẤT ĐỐI XỨNG (ACVRP)
    # ──────────────────────────────────────────────────────────────────

    def local_update_pheromone(self, start_ind: int, end_ind: int):
        """
        Local update ACS: τ_ij ← (1-ξ)·τ_ij + ξ·τ_0

        [FIX-ACVRP-1] Chỉ cập nhật chiều i→j (start→end).
        KHÔNG cập nhật chiều ngược j→i vì ma trận OSRM bất đối xứng:
        kiến di chuyển từ i đến j theo cung (i,j) có chi phí d(i,j),
        pheromone τ_ij phải độc lập với τ_ji.
        """
        self.pheromone_mat[start_ind][end_ind] = (
            (1 - self.xi) * self.pheromone_mat[start_ind][end_ind]
            + self.xi * self.init_pheromone_val
        )

    def global_update_pheromone(self, best_path: list, best_path_distance: float):
        """
        Global update: bay hơi toàn bộ rồi reinforce best path.

        [FIX-ACVRP-1] Chỉ reinforce chiều đi thực tế trong best_path.
        Với ACVRP, best_path là chuỗi node theo thứ tự xe đi thực tế;
        chiều ngược (j→i) KHÔNG được reinforce vì kiến không đi ngược.

        Công thức:
          τ_ij ← (1-ρ)·τ_ij          (bay hơi tất cả)
          τ_ij ← τ_ij + ρ/L*          (reinforce 1 chiều cho cung trên best path)
        """
        if best_path_distance <= 0:
            return

        self.pheromone_mat *= (1 - self.rho)

        delta       = self.rho / best_path_distance
        current_ind = best_path[0]
        for next_ind in best_path[1:]:
            # [FIX-ACVRP-1] Chỉ cập nhật chiều i→j
            self.pheromone_mat[current_ind][next_ind] += delta
            current_ind = next_ind

    # ──────────────────────────────────────────────────────────────────
    # Nearest Neighbor Heuristic (khởi tạo pheromone)
    # ──────────────────────────────────────────────────────────────────

    def nearest_neighbor_heuristic(self):
        """
        NNH cover TẤT CẢ customers để init_pheromone_val hợp lệ.
        Sử dụng d(i,j) đúng chiều (bất đối xứng).
        """
        index_to_visit  = list(range(1, self.node_num))
        current_index   = 0
        current_load    = 0
        travel_distance = 0.0
        travel_path     = [0]

        while index_to_visit:
            nearest = self._cal_nearest_next_index(
                index_to_visit, current_index, current_load)
            if nearest is None:
                travel_distance += self.node_dist_mat[current_index][0]
                travel_path.append(0)
                current_index = 0
                current_load  = 0
            else:
                current_load    += self.nodes[nearest].demand
                travel_distance += self.node_dist_mat[current_index][nearest]
                travel_path.append(nearest)
                current_index = nearest
                index_to_visit.remove(nearest)

        travel_distance += self.node_dist_mat[current_index][0]
        travel_path.append(0)
        vehicle_num = travel_path.count(0) - 1
        return travel_path, travel_distance, vehicle_num

    def _cal_nearest_next_index(self, index_to_visit, current_index, current_load):
        """Tìm node gần nhất thỏa capacity, theo chiều current→next."""
        nearest_ind      = None
        nearest_distance = None
        for next_index in index_to_visit:
            if current_load + self.nodes[next_index].demand > self.vehicle_capacity:
                continue
            dist = self.node_dist_mat[current_index][next_index]
            if nearest_distance is None or dist < nearest_distance:
                nearest_distance = dist
                nearest_ind      = next_index
        return nearest_ind

    @staticmethod
    def calculate_dist(node_a, node_b):
        """Khoảng cách Euclid giữa 2 node (chỉ dùng cho test)."""
        return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))


class PathMessage:
    def __init__(self, path, distance):
        """Lưu thông tin path để truyền giữa các process."""
        if path is not None:
            self.path             = copy.deepcopy(path)
            self.distance         = copy.deepcopy(distance)
            self.used_vehicle_num = self.path.count(0) - 1
        else:
            self.path             = None
            self.distance         = None
            self.used_vehicle_num = None

    def get_path_info(self):
        """Trả về (path, distance, vehicle_num)."""
        return self.path, self.distance, self.used_vehicle_num