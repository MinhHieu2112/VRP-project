import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

Route = List[int]
Solution = List[Route]

class TabuSearchSolver:
    def __init__(self,
                 distance_matrix: np.ndarray,
                 demands: Dict[int, float],
                 capacity: float,
                 max_v: int,
                 tabu_size: int = 30,
                 max_iter: int = 50_000,
                 max_no_improve: int = 1_000):
        
        # Gán các thông số cơ bản
        self.matrix = distance_matrix
        self.n = distance_matrix.shape[0]
        self.demands = demands
        self.capacity = capacity
        self.max_v = max_v
        
        # Cấu hình Tabu
        self.tabu_size = tabu_size
        self.max_iter = max_iter
        self.max_no_improve = max_no_improve
        self.tabu_list = deque(maxlen=tabu_size)

        # Cấu hình Granular Search (Cho 1600 điểm)
        self.granular_k = 30 
        self.candidates = self._build_candidate_lists()

    def _build_candidate_lists(self):
        """Tính toán sẵn danh sách k điểm gần nhất cho mỗi node."""
        candidates = {}
        for i in range(1, self.n):
            # [FIX] Dùng self.matrix đồng nhất
            sorted_indices = np.argsort(self.matrix[i])
            nearest = [j for j in sorted_indices if j != i and j != 0][:self.granular_k]
            candidates[i] = nearest
        return candidates

    # ──────────────────────────────────────────────────────────────────
    # 1. HELPERS
    # ──────────────────────────────────────────────────────────────────

    def _route_dist(self, route: Route) -> float:
        """Tính tổng khoảng cách một tuyến."""
        if len(route) <= 2: return 0
        return sum(self.matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

    def _total_dist(self, solution: Solution) -> float:
        """Tính tổng quãng đường toàn cục."""
        return sum(self._route_dist(r) for r in solution if len(r) > 2)

    def _is_valid(self, route: Route) -> bool:
        """Kiểm tra ràng buộc tải trọng."""
        demand = sum(self.demands.get(node, 0) for node in route if node != 0)
        return demand <= self.capacity

    def _copy_solution(self, solution: Solution) -> Solution:
        """Copy nông để tiết kiệm bộ nhớ."""
        return [r[:] for r in solution]
    
    def _get_node_positions(self, solution: Solution):
        """Ánh xạ Node -> (Vị trí xe, Vị trí trong xe) giúp tra cứu O(1)."""
        pos_map = {}
        for r_idx, route in enumerate(solution):
            for n_idx, node in enumerate(route):
                if node != 0:
                    pos_map[node] = (r_idx, n_idx)
        return pos_map

    # ──────────────────────────────────────────────────────────────────
    # 2. NEIGHBORHOOD OPERATORS (Sử dụng Delta Evaluation)
    # ──────────────────────────────────────────────────────────────────

    def _get_neighbors(self, solution: Solution) -> List[Tuple[Solution, tuple]]:
        """
        Tạo láng giềng thông minh bằng Granular Search.
        Tính toán độ chênh lệch quãng đường (Delta) thay vì tính lại toàn bộ.
        """
        neighbors_info = []
        node_positions = self._get_node_positions(solution)
        
        for u in range(1, self.n):
            if u not in node_positions: continue
            u_r_idx, u_pos = node_positions[u]
            u_route = solution[u_r_idx]
            
            # Các điểm liền kề u hiện tại
            prev_u, next_u = u_route[u_pos-1], u_route[u_pos+1]

            for v in self.candidates.get(u, []):
                if v not in node_positions: continue
                v_r_idx, v_pos = node_positions[v]
                if u_r_idx == v_r_idx: continue
                
                v_route = solution[v_r_idx]
                prev_v, next_v = v_route[v_pos-1], v_route[v_pos+1]

                # --- Toán tử SWAP (Đổi chỗ u và v) ---
                # Tính Delta (Chi phí tăng thêm - Chi phí mất đi)
                delta_swap = (
                    (self.matrix[prev_u, v] + self.matrix[v, next_u] - self.matrix[prev_u, u] - self.matrix[u, next_u]) +
                    (self.matrix[prev_v, u] + self.matrix[u, next_v] - self.matrix[prev_v, v] - self.matrix[v, next_v])
                )
                
                # Kiểm tra Capacity
                u_demand = self.demands[u]
                v_demand = self.demands[v]
                if (sum(self.demands[n] for n in u_route if n!=0) - u_demand + v_demand <= self.capacity and
                    sum(self.demands[n] for n in v_route if n!=0) - v_demand + u_demand <= self.capacity):
                    neighbors_info.append((delta_swap, 'swap', u, v, u_r_idx, v_r_idx, u_pos, v_pos))

                # --- Toán tử RELOCATE (Rút u chèn sau v) ---
                delta_reloc = (
                    (self.matrix[prev_u, next_u] - self.matrix[prev_u, u] - self.matrix[u, next_u]) +
                    (self.matrix[v, u] + self.matrix[u, next_v] - self.matrix[v, next_v])
                )
                if sum(self.demands[n] for n in v_route if n!=0) + u_demand <= self.capacity:
                    neighbors_info.append((delta_reloc, 'relocate', u, v, u_r_idx, v_r_idx, u_pos, v_pos))

        # Sắp xếp các nước đi tiềm năng nhất (Delta thấp nhất)
        neighbors_info.sort(key=lambda x: x[0])
        
        # Chỉ tạo ra Object Solution cho top 50 nước đi tốt nhất để tiết kiệm RAM/CPU
        final_neighbors = []
        for delta, m_type, u, v, u_idx, v_idx, u_p, v_p in neighbors_info[:50]:
            new_sol = self._copy_solution(solution)
            if m_type == 'swap':
                new_sol[u_idx][u_p], new_sol[v_idx][v_p] = v, u
                move_key = tuple(sorted([u, v]))
            else:
                new_sol[u_idx].pop(u_p)
                new_sol[v_idx].insert(v_p + 1, u)
                if len(new_sol[u_idx]) == 2: new_sol.pop(u_idx)
                move_key = (u, v_idx)
            
            final_neighbors.append((new_sol, move_key))
            
        return final_neighbors

    # ──────────────────────────────────────────────────────────────────
    # 3. MAIN SOLVE
    # ──────────────────────────────────────────────────────────────────

    def solve(self, initial_state: Solution) -> Tuple[Solution, float]:
        """Thực thi Tabu Search."""
        best_state = self._copy_solution(initial_state)
        best_dist  = self._total_dist(best_state)
        curr_state = self._copy_solution(initial_state)
        no_improve_count = 0

        print(f"Bắt đầu Tabu Search: Khoảng cách khởi tạo = {best_dist/100:.2f} km")

        for iteration in range(self.max_iter):
            if no_improve_count >= self.max_no_improve:
                break

            neighbors = self._get_neighbors(curr_state)
            if not neighbors:
                no_improve_count += 1
                continue

            best_move = None
            best_move_dist = float('inf')

            for next_state, move_key in neighbors:
                next_dist = self._total_dist(next_state)
                in_tabu = move_key in self.tabu_list

                # Aspiration Criterion: Chấp nhận nước đi Tabu nếu nó tốt hơn kỉ lục thế giới
                if next_dist < best_dist or not in_tabu:
                    if next_dist < best_move_dist:
                        best_move_dist = next_dist
                        best_move = (next_state, move_key)

            if not best_move:
                no_improve_count += 1
                continue

            # Cập nhật trạng thái hiện tại
            curr_state, move_key = best_move
            self.tabu_list.append(move_key)
            
            # Kiểm tra cải thiện
            if best_move_dist < best_dist:
                best_state = self._copy_solution(curr_state)
                best_dist = best_move_dist
                no_improve_count = 0
            else:
                no_improve_count += 1

            if iteration % 100 == 0:
                print(f"  Iter {iteration:5d} | Best: {best_dist/100:.2f} km | NoImprove: {no_improve_count}")

        print(f"Hoàn tất. Best: {best_dist/100:.2f} km")
        return best_state, best_dist