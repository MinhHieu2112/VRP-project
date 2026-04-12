"""
Algorithms/ACO/Models/cvrp_base.py
===================================
Mô hình đồ thị ACVRP với pheromone BẤT ĐỐI XỨNG (Asymmetric).

Nguyên tắc cốt lõi của ACVRP (theo báo cáo mục 1.3, 2.1.3):
  - Ma trận chi phí OSRM: d(i,j) ≠ d(j,i)  (bất đối xứng)
  - Pheromone phải phản ánh đúng chiều di chuyển: τ(i→j) ≠ τ(j→i)
  - Mọi cập nhật pheromone CHỈ theo 1 chiều i→j

Thuật toán ACS (Dorigo & Gambardella, 1997) được áp dụng:
  - Local update:  τ_ij ← (1-ξ)·τ_ij + ξ·τ_0
  - Global update: τ_ij ← (1-ρ)·τ_ij + ρ/L*  (chỉ cho cung trên best path)
  - Heuristic:     η_ij = 1/d(i,j)  (bất đối xứng tự nhiên)

FIXES SO VỚI PHIÊN BẢN CŨ:
  [FIX-ACVRP-1] local_update và global_update chỉ cập nhật 1 chiều i→j.
                 Phiên bản cũ cập nhật 2 chiều (symmetric) → sai với ACVRP.
  [FIX-ACVRP-2] seed_pheromone chỉ boost 1 chiều i→j theo route.
  [FIX-ACVRP-3] _resolve dùng realpath (thừa hưởng từ Data_loader fix).
  [FIX-NNH]     nearest_neighbor_heuristic dùng d(i,j) đúng chiều.
"""

import numpy as np
import copy
import os


class Node:
    def __init__(self, id: int, x: float, y: float, demand: float):
        """Khởi tạo node với id, toạ độ (lat/lon) và demand."""
        self.id       = id
        self.is_depot = (id == 0)
        self.x        = x
        self.y        = y
        self.demand   = demand


class CVRPGraph:
    def __init__(self, node_num: int, nodes: list, node_dist_mat: np.ndarray,
                 vehicle_capacity: int, rho: float = 0.1, xi: float = 0.01):
        """
        Khởi tạo đồ thị ACVRP với pheromone, heuristic và validation.

        Parameters
        ----------
        node_num         : Tổng số node (depot + customers)
        nodes            : List[Node]
        node_dist_mat    : Ma trận khoảng cách (float64), đơn vị nội bộ
        vehicle_capacity : Tải trọng tối đa mỗi xe
        rho              : Tốc độ bay hơi global ACS (0 < rho < 1)
        xi               : Tốc độ cập nhật local ACS  (0 < xi  < 1)
        """
        self.node_num         = node_num
        self.nodes            = nodes
        self.node_dist_mat    = node_dist_mat.astype(np.float64)
        self.vehicle_capacity = vehicle_capacity
        self.rho              = rho
        self.xi               = xi

        self._validate_inputs()

        # Khởi tạo pheromone từ NNH solution
        # τ_0 = 1 / (n * L_nnh)  — công thức chuẩn ACS
        self.nnh_travel_path, nnh_distance, _ = self.nearest_neighbor_heuristic()
        if nnh_distance <= 0:
            nnh_distance = 1.0
        self.init_pheromone_val = 1.0 / (nnh_distance * self.node_num)

        # Khởi tạo đều τ_0 cho tất cả cung
        # SAU KHI chạy, τ_ij ≠ τ_ji nhờ update 1 chiều (ACVRP)
        self.pheromone_mat = np.full(
            (self.node_num, self.node_num),
            self.init_pheromone_val,
            dtype=np.float64
        )

        # Heuristic η_ij = 1/d(i,j) — bất đối xứng tự nhiên từ OSRM
        # d(i,j) ≠ d(j,i) → η_ij ≠ η_ji ✓
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
                f"Kích thước ma trận ({mat.shape[0]}) ≠ node_num ({self.node_num})"
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

        # Kiểm tra & báo cáo tính bất đối xứng (ACVRP đặc trưng)
        diff = np.abs(mat - mat.T)
        asymmetric_count = int(np.sum(diff > 1.0))  # sai lệch > 1 đơn vị nội bộ (10m)
        if asymmetric_count > 0:
            print(f"[INFO] Ma trận ACVRP bất đối xứng: "
                  f"{asymmetric_count} cặp (i,j) có d(i,j) ≠ d(j,i) — đúng như mong đợi")
        else:
            warnings.append("Ma trận có vẻ đối xứng — kiểm tra lại dữ liệu OSRM")

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
              f"capacity={self.vehicle_capacity}, ACVRP (asymmetric distance)")

    # ──────────────────────────────────────────────────────────────────
    # Seed pheromone từ nghiệm khởi tạo
    # ──────────────────────────────────────────────────────────────────

    def seed_pheromone(self, solution: list, seed_weight: float = 2.0):
        """
        Reinforce pheromone dọc theo các cung của nghiệm seed (greedy/savings).

        [FIX-ACVRP-2] Chỉ boost 1 chiều i→j theo thứ tự xe đi thực tế.
        Không boost chiều ngược j→i vì ACVRP: τ(i→j) ≠ τ(j→i).

        Parameters
        ----------
        solution    : list của list, mỗi phần tử là một route [0, ..., 0]
        seed_weight : bội số so với init_pheromone_val (mặc định 2× = bias nhẹ)
        """
        boost = self.init_pheromone_val * seed_weight
        for route in solution:
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                # [FIX-ACVRP-2] Chỉ 1 chiều u→v
                if self.pheromone_mat[u][v] < boost:
                    self.pheromone_mat[u][v] = boost

    # ──────────────────────────────────────────────────────────────────
    # Pheromone Updates — BẤT ĐỐI XỨNG (ACVRP / ACS)
    # ──────────────────────────────────────────────────────────────────

    def local_update_pheromone(self, start_ind: int, end_ind: int):
        """
        Local update ACS sau mỗi bước di chuyển của kiến:
            τ_ij ← (1 - ξ) · τ_ij + ξ · τ_0

        [FIX-ACVRP-1] Chỉ cập nhật chiều i→j (start_ind → end_ind).
        KHÔNG cập nhật chiều ngược j→i vì:
        - d(i,j) ≠ d(j,i) trong ACVRP
        - Kiến di chuyển theo cung (i,j) → chỉ cung này nhận local evaporation
        - Cập nhật 2 chiều sẽ làm sai lệch τ_ji của chiều ngược lại
        """
        self.pheromone_mat[start_ind][end_ind] = (
            (1.0 - self.xi) * self.pheromone_mat[start_ind][end_ind]
            + self.xi * self.init_pheromone_val
        )

    def global_update_pheromone(self, best_path: list, best_path_distance: float):
        """
        Global update ACS sau mỗi iteration:
            Bước 1 — Bay hơi toàn bộ:  τ_ij ← (1 - ρ) · τ_ij  (tất cả cung)
            Bước 2 — Reinforce best:   τ_ij ← τ_ij + ρ/L*     (chỉ cung trên best path)

        [FIX-ACVRP-1] Reinforce CHỈ 1 chiều theo thứ tự thực tế trong best_path.
        KHÔNG reinforce chiều ngược vì:
        - best_path = [0, a, b, 0, c, d, 0, ...] biểu diễn thứ tự xe đi thực tế
        - Cung (a→b) được đi, còn (b→a) KHÔNG được đi trong path này
        - Reinforce (b→a) sẽ làm kiến ở b bị kéo về a sai lầm
        - Trong ACVRP, τ(a→b) ≠ τ(b→a) là cần thiết để phản ánh d(a,b) ≠ d(b,a)
        """
        if best_path_distance <= 0:
            return

        # Bước 1: Bay hơi toàn bộ (tất cả cung đồng thời)
        self.pheromone_mat *= (1.0 - self.rho)

        # Bước 2: Reinforce chỉ các cung trên best path, 1 chiều
        delta       = self.rho / best_path_distance
        current_ind = best_path[0]
        for next_ind in best_path[1:]:
            # [FIX-ACVRP-1] Chỉ cập nhật chiều i→j
            self.pheromone_mat[current_ind][next_ind] += delta
            current_ind = next_ind

    # ──────────────────────────────────────────────────────────────────
    # Nearest Neighbor Heuristic — khởi tạo τ_0
    # ──────────────────────────────────────────────────────────────────

    def nearest_neighbor_heuristic(self):
        """
        NNH cover TẤT CẢ customers để init_pheromone_val hợp lệ.

        Sử dụng d(i,j) đúng chiều (bất đối xứng) — không dùng d(j,i).
        Khi không có node nào vừa capacity → về depot mở tuyến mới.
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
                # Không còn node vừa capacity → về depot
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

        # Kết thúc tại depot
        travel_distance += self.node_dist_mat[current_index][0]
        travel_path.append(0)

        vehicle_num = travel_path.count(0) - 1
        return travel_path, travel_distance, vehicle_num

    def _cal_nearest_next_index(self, index_to_visit: list,
                                 current_index: int,
                                 current_load: int):
        """
        Tìm node gần nhất thỏa capacity theo chiều current → next.
        Sử dụng d(current, next) — bất đối xứng.
        """
        nearest_ind      = None
        nearest_distance = float('inf')

        for next_index in index_to_visit:
            if current_load + self.nodes[next_index].demand > self.vehicle_capacity:
                continue
            dist = self.node_dist_mat[current_index][next_index]
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_ind      = next_index

        return nearest_ind

    @staticmethod
    def calculate_dist(node_a: Node, node_b: Node) -> float:
        """Khoảng cách Euclid giữa 2 node (chỉ dùng cho unit test)."""
        return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))


class PathMessage:
    def __init__(self, path, distance):
        """Lưu thông tin path để truyền giữa các process (multiprocessing)."""
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