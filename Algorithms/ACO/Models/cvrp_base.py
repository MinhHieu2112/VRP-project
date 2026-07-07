# File định nghĩa các lớp mô hình đồ thị CVRP và thực thể Node phục vụ thuật toán ACO.
import numpy as np
import copy
import os


class Node:
    """Đại diện cho một nút trong bài toán VRP (depot hoặc khách hàng)."""

    def __init__(self, id: int, x: float, y: float, demand: float):
        # Khởi tạo thông tin của một node gồm vị trí và nhu cầu vận chuyển.
        self.id       = id
        self.is_depot = (id == 0)
        self.x        = x
        self.y        = y
        self.demand   = demand


class CVRPGraph:
    """Đồ thị bài toán CVRP lưu trữ ma trận khoảng cách và pheromone cho ACO."""

    def __init__(self, node_num: int, nodes: list, node_dist_mat: np.ndarray,
                 vehicle_capacity: int, rho: float = 0.1, xi: float = 0.01):
        # Khởi tạo đồ thị, ma trận khoảng cách, ma trận pheromone và thông tin heuristic.
        self.node_num         = node_num
        self.nodes            = nodes
        self.node_dist_mat    = node_dist_mat.astype(np.float32)
        self.vehicle_capacity = vehicle_capacity
        self.rho              = rho
        self.xi               = xi

        self._validate_inputs()

        self.nnh_travel_path, nnh_distance, _ = self.nearest_neighbor_heuristic()
        if nnh_distance <= 0:
            nnh_distance = 1.0
        self.init_pheromone_val = 1.0 / (nnh_distance * self.node_num)

        self.pheromone_mat = np.full(
            (self.node_num, self.node_num),
            self.init_pheromone_val,
            dtype=np.float32
        )

        with np.errstate(divide='ignore', invalid='ignore'):
            self.heuristic_info_mat = np.where(
                self.node_dist_mat > 0,
                1.0 / self.node_dist_mat,
                0.0
            ).astype(np.float32)
        np.fill_diagonal(self.heuristic_info_mat, 0.0)

    def _validate_inputs(self):
        # Kiểm tra tính hợp lệ của dữ liệu đầu vào trước khi chạy thuật toán.
        errors   = []
        warnings = []
        mat      = self.node_dist_mat

        if mat.shape[0] != mat.shape[1]:
            errors.append(f"Ma trận không vuông: {mat.shape}")
        if mat.shape[0] != self.node_num:
            errors.append(f"Kích thước ma trận ({mat.shape[0]}) ≠ node_num ({self.node_num})")
        if np.any(np.isnan(mat)):
            errors.append("Ma trận chứa NaN")
        if np.any(np.isinf(mat)):
            errors.append("Ma trận chứa Inf")

        NEG_TOLERANCE = -0.01
        neg_mask = mat < 0
        if np.any(neg_mask):
            min_neg   = mat[neg_mask].min()
            neg_count = int(neg_mask.sum())
            if min_neg >= NEG_TOLERANCE:
                warnings.append(f"{neg_count} giá trị âm nhỏ (min={min_neg:.6f}) → clip về 0")
                self.node_dist_mat = np.clip(self.node_dist_mat, 0.0, None)
            else:
                errors.append(f"{neg_count} giá trị âm lớn (min={min_neg:.4f})")

        diff = np.abs(mat - mat.T)
        asymmetric_count = int(np.sum(diff > 1.0))
        if asymmetric_count > 0:
            print(f"[INFO] Ma trận ACVRP bất đối xứng: "
                  f"{asymmetric_count} cặp (i,j) có d(i,j) ≠ d(j,i)")
        else:
            warnings.append("Ma trận có vẻ đối xứng — kiểm tra lại dữ liệu OSRM")

        infeasible = [
            (node.id, node.demand)
            for node in self.nodes
            if not node.is_depot and node.demand > self.vehicle_capacity
        ]
        if infeasible:
            errors.append(
                f"{len(infeasible)} node có demand > capacity: {infeasible[:5]}")

        for w in warnings:
            print(f"[WARN] {w}")
        if errors:
            raise ValueError("Dữ liệu không hợp lệ:\n" + "\n".join(f"  - {e}" for e in errors))

        print(f"[OK] Validation passed: {self.node_num} nodes, "
              f"capacity={self.vehicle_capacity}, ACVRP (asymmetric)")

    def seed_pheromone(self, solution: list, seed_weight: float = 2.0):
        # Khởi tạo pheromone ban đầu dựa trên nghiệm mốc nhằm định hướng kiến.
        boost = np.float32(self.init_pheromone_val * seed_weight)
        for route in solution:
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                if self.pheromone_mat[u][v] < boost:
                    self.pheromone_mat[u][v] = boost

    def local_update_pheromone(self, start_ind: int, end_ind: int):
        # Cập nhật pheromone cục bộ trên cạnh mà kiến vừa đi qua.
        self.pheromone_mat[start_ind][end_ind] = (
            (1.0 - self.xi) * self.pheromone_mat[start_ind][end_ind]
            + self.xi * self.init_pheromone_val
        )

    def global_update_pheromone(self, best_path: list, best_path_distance: float):
        # Cập nhật pheromone toàn cục theo đường đi tốt nhất của vòng lặp.
        self.global_update_pheromone_sparse(best_path, best_path_distance)

    def global_update_pheromone_sparse(self, best_path: list, best_path_distance: float):
        # Cập nhật pheromone thưa thớt chỉ trên các cạnh của best path.
        if best_path_distance <= 0:
            return

        delta = np.float32(self.rho / best_path_distance)
        one_minus_rho = np.float32(1.0 - self.rho)
        pm = self.pheromone_mat

        current_ind = best_path[0]
        for next_ind in best_path[1:]:
            pm[current_ind][next_ind] = (
                one_minus_rho * pm[current_ind][next_ind] + delta
            )
            current_ind = next_ind

    def nearest_neighbor_heuristic(self):
        # Xây dựng nghiệm ban đầu theo heuristic láng giềng gần nhất, sử dụng NumPy masked argmin để đạt O(N) mỗi bước.
        n               = self.node_num
        demands         = np.array([self.nodes[i].demand for i in range(n)], dtype=np.float32)
        visited         = np.zeros(n, dtype=bool)
        visited[0]      = True
        current_index   = 0
        current_load    = 0.0
        travel_distance = 0.0
        travel_path     = [0]
        remaining       = n - 1  # số khách hàng chưa thăm

        while remaining > 0:
            row = self.node_dist_mat[current_index].copy()
            # Che các node đã thăm và node vi phạm tải trọng
            feasible_mask = (~visited) & ((current_load + demands) <= self.vehicle_capacity)

            if not np.any(feasible_mask):
                # Quay về depot khi không còn node khả thi
                travel_distance += self.node_dist_mat[current_index][0]
                travel_path.append(0)
                current_index = 0
                current_load  = 0.0
            else:
                row[~feasible_mask] = np.inf
                nearest = int(np.argmin(row))
                current_load    += demands[nearest]
                travel_distance += self.node_dist_mat[current_index][nearest]
                travel_path.append(nearest)
                visited[nearest] = True
                current_index    = nearest
                remaining       -= 1

        travel_distance += self.node_dist_mat[current_index][0]
        travel_path.append(0)

        vehicle_num = travel_path.count(0) - 1
        return travel_path, float(travel_distance), vehicle_num

    def _cal_nearest_next_index(self, index_to_visit, current_index, current_load):
        # Tìm node gần nhất còn lại thỏa mãn ràng buộc tải trọng từ vị trí hiện tại.
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
        # Tính khoảng cách Euclidean giữa hai node.
        return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))


class PathMessage:
    """Cấu trúc thông điệp chứa thông tin đường đi tốt nhất."""

    def __init__(self, path, distance):
        # Lưu trữ thông tin về đường đi và số lượng xe sử dụng.
        if path is not None:
            self.path             = copy.deepcopy(path)
            self.distance         = copy.deepcopy(distance)
            self.used_vehicle_num = self.path.count(0) - 1
        else:
            self.path             = None
            self.distance         = None
            self.used_vehicle_num = None

    def get_path_info(self):
        # Trả về bộ thông tin gồm đường đi, khoảng cách và số xe sử dụng.
        return self.path, self.distance, self.used_vehicle_num