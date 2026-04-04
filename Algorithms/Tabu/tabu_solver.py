import numpy as np
import copy
import time


class TabuSearchSolver:
    """
    Giải bài toán ACVRP (Asymmetric Capacitated Vehicle Routing Problem)
    bằng thuật toán Tabu Search.

    Bài toán: Tìm tập hợp các tuyến đường cho K xe xuất phát từ depot (node 0),
    phục vụ tất cả khách hàng, với tổng demand mỗi tuyến không vượt capacity,
    sao cho tổng quãng đường di chuyển là nhỏ nhất.
    """

    def __init__(self, distance_matrix, demands, capacity, max_v,
                 tabu_size=30, max_iter=10000, max_runtime=180):
        """
        Khởi tạo solver.

        Args:
            distance_matrix: Ma trận khoảng cách ASYMMETRIC (n x n),
                             matrix[i][j] != matrix[j][i] trong trường hợp tổng quát.
            demands:         Dict {node_id: demand} — nhu cầu của từng khách hàng.
                             Depot (node 0) có demand = 0.
            capacity:        Sức chứa tối đa của mỗi xe.
            max_v:           Số xe tối đa được phép sử dụng.
            tabu_size:       Độ dài tối đa của danh sách tabu.
            max_iter:        Số vòng lặp tối đa.
            max_runtime:     Giới hạn thời gian chạy (giây).
        """
        self.matrix = distance_matrix
        self.demands = demands          # ← THÊM: lưu demand thực tế từng node
        self.capacity = capacity
        self.max_v = max_v
        self.tabu_size = tabu_size
        self.max_iter = max_iter
        self.max_runtime = max_runtime
        self.tabu_list = []             # Danh sách các move đã thực hiện gần đây

    # ─────────────────────────────────────────────
    # TÍNH TOÁN CHI PHÍ
    # ─────────────────────────────────────────────

    def _route_dist(self, route):
        """
        Tính tổng khoảng cách của MỘT tuyến đường.
        Lưu ý: Ma trận ASYMMETRIC nên matrix[i][j] != matrix[j][i].

        Args:
            route: List các node, ví dụ [0, 3, 7, 2, 0]
        Returns:
            Tổng khoảng cách (float)
        """
        total = 0.0
        for i in range(len(route) - 1):
            total += self.matrix[route[i], route[i + 1]]
        return total

    def _total_dist(self, state):
        """
        Tính tổng khoảng cách của TOÀN BỘ lời giải (tất cả các tuyến).

        Args:
            state: List các route, mỗi route là list node bắt đầu và kết thúc bằng 0.
        Returns:
            Tổng khoảng cách (float)
        """
        return sum(self._route_dist(r) for r in state if len(r) > 2)

    def _route_demand(self, route):
        """
        Tính tổng demand của một tuyến đường (không tính depot).

        Args:
            route: List các node
        Returns:
            Tổng demand (int/float)
        """
        return sum(self.demands.get(node, 0) for node in route if node != 0)

    def _is_feasible(self, route):
        """
        Kiểm tra tuyến đường có khả thi không:
        - Tổng demand không vượt quá capacity của xe.

        FIX: Phiên bản cũ kiểm tra số lượng node (len-2) thay vì tổng demand thực,
        dẫn đến bỏ sót vi phạm khi các node có demand > 1.

        Args:
            route: List các node
        Returns:
            True nếu khả thi, False nếu vi phạm
        """
        return self._route_demand(route) <= self.capacity

    # ─────────────────────────────────────────────
    # CÁC TOÁN TỬ SINH LÁNG GIỀNG (MOVE OPERATORS)
    # ─────────────────────────────────────────────

    def _move_relocate(self, state):
        """
        Toán tử RELOCATE (inter-route):
        Lấy một khách hàng từ tuyến v1 và chèn vào vị trí tốt nhất trong tuyến v2.

        Đây là toán tử hiệu quả nhất cho VRP — giải quyết vấn đề
        tuyến đường quá dài / quá ngắn do mất cân bằng tải.

        Returns:
            List các (new_state, move_key) — move_key dùng cho tabu list
        """
        neighbors = []
        n_routes = len(state)

        for _ in range(30):
            v1 = np.random.randint(0, n_routes)
            v2 = np.random.randint(0, n_routes)
            if v1 == v2 or len(state[v1]) <= 3:
                # v1 chỉ còn 1 khách hàng — không lấy ra được
                continue

            # Chọn ngẫu nhiên 1 khách hàng từ v1 (không phải depot)
            p1 = np.random.randint(1, len(state[v1]) - 1)
            customer = state[v1][p1]

            # Thử chèn customer vào từng vị trí của v2
            for p2 in range(1, len(state[v2])):
                new_state = copy.deepcopy(state)
                new_state[v1].pop(p1)
                new_state[v2].insert(p2, customer)

                if self._is_feasible(new_state[v2]):
                    move_key = ('relocate', v1, v2, customer, p2)
                    neighbors.append((new_state, move_key))
                    break  # Chỉ lấy 1 vị trí chèn tốt nhất (greedy)

        return neighbors

    def _move_swap(self, state):
        """
        Toán tử SWAP (inter-route):
        Đổi chỗ một khách hàng ở tuyến v1 với một khách hàng ở tuyến v2.

        FIX: Phiên bản cũ kiểm tra capacity bằng len(route)-2 (số node),
        đúng khi demand đồng nhất = 1, nhưng SAI với demand không đồng nhất.
        Nay thay bằng _is_feasible() kiểm tra tổng demand thực.

        Returns:
            List các (new_state, move_key)
        """
        neighbors = []

        for _ in range(20):
            v1, v2 = np.random.choice(len(state), 2, replace=False)
            if len(state[v1]) <= 2 or len(state[v2]) <= 2:
                continue

            p1 = np.random.randint(1, len(state[v1]) - 1)
            p2 = np.random.randint(1, len(state[v2]) - 1)

            new_state = copy.deepcopy(state)
            # Hoán đổi 2 khách hàng giữa 2 tuyến
            new_state[v1][p1], new_state[v2][p2] = new_state[v2][p2], new_state[v1][p1]

            # Kiểm tra DEMAND THỰC TẾ (không phải số node)
            if self._is_feasible(new_state[v1]) and self._is_feasible(new_state[v2]):
                c1 = state[v1][p1]
                c2 = state[v2][p2]
                # Chuẩn hóa move_key để tránh trùng lặp (v1,c1) vs (v2,c2)
                move_key = ('swap', min(c1, c2), max(c1, c2))
                neighbors.append((new_state, move_key))

        return neighbors

    def _move_2opt_intra(self, state):
        """
        Toán tử 2-OPT (intra-route):
        Đảo ngược một đoạn con bên trong cùng một tuyến đường.

        Cần thiết cho ACVRP (asymmetric): đảo chiều đoạn [i..j] có thể giảm
        chi phí do tận dụng cạnh thuận chiều trong ma trận không đối xứng.

        Returns:
            List các (new_state, move_key)
        """
        neighbors = []

        for _ in range(20):
            v = np.random.randint(0, len(state))
            route = state[v]
            if len(route) < 5:
                # Cần ít nhất 3 khách hàng để 2-opt có ý nghĩa
                continue

            i = np.random.randint(1, len(route) - 2)
            j = np.random.randint(i + 1, len(route) - 1)

            new_state = copy.deepcopy(state)
            # Đảo ngược đoạn [i, j] trong tuyến
            new_state[v][i:j + 1] = new_state[v][i:j + 1][::-1]

            # 2-opt không thay đổi load → không cần kiểm tra feasibility
            move_key = ('2opt', v, i, j)
            neighbors.append((new_state, move_key))

        return neighbors

    def _get_neighbors(self, state):
        """
        Tổng hợp láng giềng từ TẤT CẢ các toán tử.

        FIX: Phiên bản cũ chỉ dùng swap → dễ kẹt local optimum.
        Nay kết hợp 3 toán tử: relocate + swap + 2-opt.

        Returns:
            List các (new_state, move_key), đã sắp xếp theo chi phí tăng dần
        """
        neighbors = []
        neighbors += self._move_relocate(state)
        neighbors += self._move_swap(state)
        neighbors += self._move_2opt_intra(state)

        # Sắp xếp để ưu tiên láng giềng tốt nhất
        neighbors.sort(key=lambda x: self._total_dist(x[0]))
        return neighbors

    # ─────────────────────────────────────────────
    # VÒNG LẶP CHÍNH TABU SEARCH
    # ─────────────────────────────────────────────

    def solve(self, initial_state):
        """
        Thực thi vòng lặp Tabu Search chính.

        Chiến lược:
        - Mỗi vòng lặp: sinh láng giềng → chọn move tốt nhất không bị cấm
          (hoặc vi phạm tabu nhưng thỏa aspiration criterion).
        - Aspiration criterion: Chấp nhận move bị cấm nếu nó cho kết quả
          TỐT HƠN best đã biết (tránh bỏ lỡ lời giải tối ưu).
        - Dừng khi hết max_iter hoặc vượt max_runtime.

        Args:
            initial_state: Lời giải ban đầu — list các route

        Returns:
            (best_state, best_dist): Lời giải tốt nhất tìm được và chi phí tương ứng
        """
        best_state = copy.deepcopy(initial_state)
        best_dist = self._total_dist(best_state)
        curr_state = copy.deepcopy(initial_state)

        start_time = time.time()

        for iteration in range(self.max_iter):

            # ── Kiểm tra giới hạn thời gian ──
            elapsed = time.time() - start_time
            if elapsed > self.max_runtime:
                print(f"--- Đã đạt giới hạn thời gian {self.max_runtime}s. Dừng tại vòng {iteration}. ---")
                break

            neighbors = self._get_neighbors(curr_state)
            if not neighbors:
                continue

            # ── Chọn move tốt nhất thỏa tabu + aspiration ──
            move_accepted = False
            for next_state, move_key in neighbors:
                next_dist = self._total_dist(next_state)
                in_tabu = move_key in self.tabu_list

                # Aspiration criterion: chấp nhận dù bị cấm nếu cải thiện best
                if not in_tabu or next_dist < best_dist:
                    curr_state = next_state

                    if next_dist < best_dist:
                        best_state = copy.deepcopy(next_state)
                        best_dist = next_dist

                    # Cập nhật tabu list (FIFO queue)
                    self.tabu_list.append(move_key)
                    if len(self.tabu_list) > self.tabu_size:
                        self.tabu_list.pop(0)

                    move_accepted = True
                    break

            # ── Log tiến trình mỗi 100 vòng ──
            if iteration % 100 == 0:
                print(f"  Iter {iteration:5d} | Time: {elapsed:6.1f}s | "
                      f"Curr: {self._total_dist(curr_state):8.2f} | "
                      f"Best: {best_dist:8.2f}")

        print(f"\n[Tabu Search] Hoàn tất. Best distance = {best_dist:.2f}")
        return best_state, best_dist