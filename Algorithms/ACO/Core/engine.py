# Lớp điều phối chính thực thi giải thuật tối ưu hóa Ant Colony Optimization (ACO).
import numpy as np
import random
import time
import math
from Models.cvrp_base import CVRPGraph
from Core.ant import Ant

KM_SCALE = 100 

# Số láng giềng tối đa trong candidate list
CANDIDATE_K = 20

class BasicACO:
    def __init__(self,
                 graph: CVRPGraph,
                 ants_num: int         = 10,
                 max_iter: int         = 50_000,
                 alpha: float          = 1.0,
                 beta: float           = 2.0,
                 q0: float             = 0.9,
                 no_improve_limit: int = 500):
        # Khởi tạo các thông số, cấu trúc pheromone và danh sách kiến cố định.
        self.graph            = graph
        self.ants_num         = ants_num
        self.max_iter         = max_iter
        self.alpha            = alpha
        self.beta             = beta
        self.q0               = q0
        self.no_improve_limit = no_improve_limit

        self.best_path_distance: float | None = None
        self.best_path: list | None           = None
        self.best_vehicle_num: int | None     = None

        # Cache demands array để vectorized check
        self._demands = np.array(
            [graph.nodes[i].demand for i in range(graph.node_num)],
            dtype=np.float32
        )

        # [PERF-2] Precompute candidate lists (top-K láng giềng gần nhất cho mỗi node)
        self._candidate_lists = self._build_candidate_lists(k=CANDIDATE_K)
        
        # Khởi tạo mảng các kiến cố định phục vụ Object Pooling
        self.ants = [Ant(self.graph) for _ in range(self.ants_num)]
        
        print(f"[ACO] Candidate lists built: top-{CANDIDATE_K} per node, "
              f"n={graph.node_num}, ants={ants_num}")

    # ──────────────────────────────────────────────────────────────────
    # Precompute candidate lists
    # ──────────────────────────────────────────────────────────────────

    def _build_candidate_lists(self, k: int) -> np.ndarray:
        # Xây dựng danh sách candidate list chứa k láng giềng gần nhất cho mỗi node.
        n = self.graph.node_num
        dist_mat = self.graph.node_dist_mat  # (n, n)
        candidate_lists = np.zeros((n, k), dtype=np.int32)

        for i in range(n):
            row = dist_mat[i].copy()
            row[i] = np.inf  # loại bỏ self
            row[0] = np.inf  # loại bỏ depot (depot được xử lý riêng)
            nearest = np.argpartition(row, min(k, n - 2))[:min(k, n - 2)]
            nearest = nearest[np.argsort(row[nearest])]
            candidate_lists[i, :len(nearest)] = nearest
            if len(nearest) < k:
                candidate_lists[i, len(nearest):] = -1  # padding

        return candidate_lists

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def run_basic_aco(self):
        # Chạy giải thuật ACO để tìm kiếm lộ trình tối ưu và trả về kết quả.
        self._basic_aco()
        self.best_vehicle_num = self._count_valid_vehicles(self.best_path)
        return self.best_path, self.best_path_distance, self.best_vehicle_num

    # ──────────────────────────────────────────────────────────────────
    # Core loop
    # ──────────────────────────────────────────────────────────────────

    def _basic_aco(self):
        # Vòng lặp tối ưu chính của giải thuật ACO.
        start_time       = time.time()
        no_improve_count = 0
        q0_current       = self.q0
        half_limit       = max(self.no_improve_limit // 2, 50)

        for iteration in range(self.max_iter):

            if no_improve_count >= self.no_improve_limit:
                print(f"[ACO] Dừng sớm tại vòng {iteration}: "
                      f"{no_improve_count} vòng không cải thiện "
                      f"(ngưỡng={self.no_improve_limit}).")
                break

            if no_improve_count == half_limit:
                q0_current = max(self.q0 - 0.2, 0.5)
                print(f"  [ACO] Exploration boost tại iter {iteration}: "
                      f"q0 {self.q0:.2f}→{q0_current:.2f}")
            elif no_improve_count == 0 and q0_current != self.q0:
                q0_current = self.q0

            # Sử dụng Object Pooling: Reset trạng thái của các con kiến thay vì tạo mới
            for ant in self.ants:
                ant.reset()
                self._construct_solution(ant, q0_current)

            improved = False
            for ant in self.ants:
                if (self.best_path is None
                        or ant.total_travel_distance < self.best_path_distance):
                    self.best_path          = ant.travel_path[:]
                    self.best_path_distance = ant.total_travel_distance
                    self.best_vehicle_num   = self._count_valid_vehicles(self.best_path)
                    improved = True
                    km = self.best_path_distance / KM_SCALE
                    print(f"[iter {iteration:5d}] Improved: "
                          f"{self.best_path_distance:.0f} units ({km:.2f} km), "
                          f"vehicles={self.best_vehicle_num}")

            no_improve_count = 0 if improved else no_improve_count + 1

            self.graph.global_update_pheromone_sparse(
                self.best_path, self.best_path_distance)

            if iteration % 50 == 0 and iteration > 0:
                elapsed = time.time() - start_time
                km = self.best_path_distance / KM_SCALE
                print(f"  [iter {iteration:5d}] elapsed={elapsed:.1f}s | "
                      f"NoImprove={no_improve_count}/{self.no_improve_limit} | "
                      f"best={km:.2f} km | q0_eff={q0_current:.2f}")

        elapsed = time.time() - start_time
        km = self.best_path_distance / KM_SCALE
        print(f"\n[DONE] {self.best_path_distance:.0f} units ({km:.2f} km), "
              f"vehicles={self.best_vehicle_num}, time={elapsed:.2f}s")

    # ──────────────────────────────────────────────────────────────────
    # Xây nghiệm cho một kiến
    # ──────────────────────────────────────────────────────────────────
    def _construct_solution(self, ant: Ant, q0: float):
        # Xây dựng lộ trình hoàn chỉnh cho một kiến dựa trên tri thức heuristic và pheromone.
        n_customers        = self.graph.node_num - 1
        max_customer_steps = n_customers
        customer_steps     = 0

        local_update_batch = []

        while not ant.index_to_visit_empty():

            if customer_steps > max_customer_steps:
                remaining = ant.index_to_visit
                print(f"[WARN] customer_steps ({customer_steps}) > max ({max_customer_steps}), "
                      f"force-visit {len(remaining)} node còn lại")
                ant.force_visit_remaining(remaining)
                break

            feasible = self._get_feasible_candidates(ant)

            if not feasible:
                if not ant.is_at_depot():
                    prev = ant.current_index
                    ant.move_to_next_index(0)
                    local_update_batch.append((prev, 0))
                else:
                    print("[ERROR] Ant kẹt tại depot — demand đơn lẻ > capacity.")
                    ant.force_visit_remaining(ant.index_to_visit)
                    break
            else:
                prev       = ant.current_index
                next_index = self.select_next_index(ant, feasible, q0)
                ant.move_to_next_index(next_index)
                local_update_batch.append((prev, next_index))
                if next_index != 0:
                    customer_steps += 1

        if not ant.is_at_depot():
            prev = ant.current_index
            ant.move_to_next_index(0)
            local_update_batch.append((prev, 0))

        self._apply_local_update_batch(local_update_batch)

    def _fallback_feasible(self, ant: Ant) -> list:
        # Quét tìm kiếm các node khả thi tiếp theo trên toàn bộ đồ thị sử dụng list boolean.
        load = ant.vehicle_load
        capacity = self.graph.vehicle_capacity
        demands = self._demands
        visited = ant.visited

        feasible = []
        for idx in range(1, len(visited)):
            if not visited[idx] and (load + demands[idx]) <= capacity:
                feasible.append(idx)
        return feasible

    def _get_feasible_candidates(self, ant: Ant) -> list:
        # Lấy danh sách các node khả thi xung quanh kiến, ưu tiên duyệt trên candidate list.
        current = ant.current_index
        visited = ant.visited
        load = ant.vehicle_load
        capacity = self.graph.vehicle_capacity
        demands = self._demands

        # Bước 1: Xét candidate list của node hiện tại
        candidates = self._candidate_lists[current]
        feasible_from_candidates = []
        for nb in candidates:
            if nb == -1:
                break  # padding
            node = int(nb)
            if not visited[node] and (load + demands[node]) <= capacity:
                feasible_from_candidates.append(node)

        if feasible_from_candidates:
            return feasible_from_candidates

        # Bước 2: Fallback khi candidate list không chứa ứng viên hợp lệ
        return self._fallback_feasible(ant)

    def _apply_local_update_batch(self, batch: list):
        # Cập nhật pheromone cục bộ cho các cạnh mà kiến đã đi qua trong lượt đi.
        xi  = self.graph.xi
        tau0 = self.graph.init_pheromone_val
        pm   = self.graph.pheromone_mat
        for (i, j) in batch:
            pm[i][j] = (1.0 - xi) * pm[i][j] + xi * tau0

    # ──────────────────────────────────────────────────────────────────
    # Chọn node tiếp theo — ACS transition rule
    # ──────────────────────────────────────────────────────────────────
    def select_next_index(self, ant: Ant, feasible_nodes: list, q0: float) -> int:
        # Chọn node tiếp theo dựa trên quy tắc chuyển trạng thái của kiến (ACS transition rule) bằng Python thuần để tránh Numpy overhead.
        current_index = ant.current_index
        pheromone_row = self.graph.pheromone_mat[current_index]
        heuristic_row = self.graph.heuristic_info_mat[current_index]

        alpha = self.alpha
        beta = self.beta

        n_feasible = len(feasible_nodes)
        scores = [0.0] * n_feasible
        score_sum = 0.0

        for idx, node in enumerate(feasible_nodes):
            p = pheromone_row[node]
            h = heuristic_row[node]

            score_p = p if alpha == 1.0 else (p ** alpha)
            score_h = (h * h) if beta == 2.0 else (h ** beta)
            score = score_p * score_h

            if not math.isfinite(score) or score < 0:
                score = 0.0

            scores[idx] = score
            score_sum += score

        if score_sum <= 0:
            return random.choice(feasible_nodes)

        if random.random() < q0:
            best_idx = 0
            best_score = scores[0]
            for idx in range(1, n_feasible):
                if scores[idx] > best_score:
                    best_score = scores[idx]
                    best_idx = idx
            return feasible_nodes[best_idx]
        else:
            r = random.random() * score_sum
            accumulator = 0.0
            for idx, score in enumerate(scores):
                accumulator += score
                if accumulator >= r:
                    return feasible_nodes[idx]
            return feasible_nodes[-1]

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _roulette_wheel(candidates: list, probs: np.ndarray) -> int:
        # Chọn một phần tử ngẫu nhiên dựa trên mảng phân phối xác suất tích lũy.
        probs = np.clip(probs, 0, None)
        total = probs.sum()
        if total <= 0:
            return random.choice(candidates)
        chosen_idx = np.random.choice(len(candidates), p=probs / total)
        return candidates[chosen_idx]

    @staticmethod
    def _count_valid_vehicles(path: list) -> int:
        # Đếm số lượng xe thực tế đang hoạt động trong lộ trình thu được.
        count, has_customer = 0, False
        for node in path[1:]:
            if node == 0:
                if has_customer:
                    count += 1
                has_customer = False
            else:
                has_customer = True
        if has_customer:
            count += 1
        return count