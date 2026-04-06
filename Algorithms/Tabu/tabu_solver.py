"""
Algorithms/Tabu/tabu_solver.py  (refactored)
=============================================
TabuSearchSolver — Tabu Search cho CVRP.

Sửa lỗi so với phiên bản cũ:
  [FIX-1] copy.deepcopy trong neighbor generation: cực kỳ chậm với 200 xe.
          Thay bằng shallow copy có mục tiêu (chỉ copy route bị sửa).
  [FIX-2] _move_relocate: sau khi xóa node khỏi route rỗng không đóng
          [0,0] — giờ kiểm tra và giữ nguyên format [0, ..., 0].
  [FIX-3] Khởi tạo nghiệm dùng init_strategies (tách khỏi main_tabu.py).
"""

import numpy as np
import copy
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
        """
        Khởi tạo Tabu Search.

        Parameters
        ----------
        distance_matrix : Ma trận khoảng cách (mét, int).
        demands         : Dict {node_id: demand}.
        capacity        : Sức chứa mỗi xe.
        max_v           : Số xe tối đa.
        tabu_size       : Độ dài danh sách tabu.
        max_iter        : Vòng lặp tối đa (safety cap).
        max_no_improve  : Dừng sớm sau n vòng không cải thiện.
        """
        self.matrix       = distance_matrix
        self.demands      = demands
        self.capacity     = capacity
        self.max_v        = max_v
        self.tabu_size    = tabu_size
        self.max_iter     = max_iter
        self.max_no_improve = max_no_improve
        self.tabu_list    = deque(maxlen=tabu_size)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _route_dist(self, route: Route) -> float:
        """Tính tổng khoảng cách một route."""
        return sum(self.matrix[route[i], route[i + 1]]
                   for i in range(len(route) - 1))

    def _total_dist(self, solution: Solution) -> float:
        """Tính tổng khoảng cách toàn bộ nghiệm."""
        return sum(self._route_dist(r) for r in solution if len(r) > 2)

    def _route_demand(self, route: Route) -> float:
        """Tính tổng demand của một route."""
        return sum(self.demands.get(node, 0) for node in route if node != 0)

    def _is_feasible(self, route: Route) -> bool:
        """Kiểm tra capacity constraint của một route."""
        return self._route_demand(route) <= self.capacity

    def _copy_solution(self, solution: Solution) -> Solution:
        """
        [FIX-1] Copy nhẹ: chỉ copy list các route, không deepcopy toàn bộ.
        Đủ an toàn vì ta chỉ modify route cụ thể trong neighbor generation.
        """
        return [r[:] for r in solution]

    # ──────────────────────────────────────────────────────────────────
    # Neighborhood operators
    # ──────────────────────────────────────────────────────────────────

    def _move_relocate(self, solution: Solution) -> List[Tuple[Solution, tuple]]:
        """
        [FIX-1] Operator RELOCATE: di chuyển 1 khách từ route này sang route khác.
        Dùng _copy_solution thay deepcopy để tăng tốc độ đáng kể.
        [FIX-2] Route trở nên rỗng [0,0] được giữ nguyên (không xóa tại đây);
        _total_dist bỏ qua route len <= 2.
        """
        neighbors = []
        n_routes  = len(solution)

        for _ in range(30):
            v1 = np.random.randint(0, n_routes)
            v2 = np.random.randint(0, n_routes)
            if v1 == v2 or len(solution[v1]) <= 3:
                continue

            p1       = np.random.randint(1, len(solution[v1]) - 1)
            customer = solution[v1][p1]

            # Thử tất cả vị trí trong route v2
            for p2 in range(1, len(solution[v2])):
                # [FIX-1] copy nhẹ
                new_sol = self._copy_solution(solution)
                new_sol[v1].pop(p1)
                new_sol[v2].insert(p2, customer)

                if self._is_feasible(new_sol[v2]):
                    move_key = ('relocate', v1, v2, customer, p2)
                    neighbors.append((new_sol, move_key))
                    break  # Chỉ lấy vị trí đầu tiên hợp lệ

        return neighbors

    def _move_swap(self, solution: Solution) -> List[Tuple[Solution, tuple]]:
        """
        Operator SWAP: đổi chỗ 2 khách hàng giữa 2 route khác nhau.
        Kiểm tra capacity cả 2 route sau swap.
        """
        neighbors = []

        for _ in range(20):
            v1, v2 = np.random.choice(len(solution), 2, replace=False)
            if len(solution[v1]) <= 2 or len(solution[v2]) <= 2:
                continue

            p1 = np.random.randint(1, len(solution[v1]) - 1)
            p2 = np.random.randint(1, len(solution[v2]) - 1)

            # [FIX-1] copy nhẹ
            new_sol = self._copy_solution(solution)
            new_sol[v1][p1], new_sol[v2][p2] = new_sol[v2][p2], new_sol[v1][p1]

            if (self._is_feasible(new_sol[v1])
                    and self._is_feasible(new_sol[v2])):
                c1, c2   = solution[v1][p1], solution[v2][p2]
                move_key = ('swap', min(c1, c2), max(c1, c2))
                neighbors.append((new_sol, move_key))

        return neighbors

    def _move_2opt_intra(self, solution: Solution) -> List[Tuple[Solution, tuple]]:
        """
        Operator 2-OPT nội tuyến: đảo ngược đoạn giữa 2 vị trí trong cùng route.
        Không cần kiểm tra capacity (chỉ đổi thứ tự, không thay đổi thành phần).
        """
        neighbors = []

        for _ in range(20):
            v     = np.random.randint(0, len(solution))
            route = solution[v]
            if len(route) < 5:
                continue

            i = np.random.randint(1, len(route) - 2)
            j = np.random.randint(i + 1, len(route) - 1)

            # [FIX-1] copy nhẹ
            new_sol       = self._copy_solution(solution)
            new_sol[v][i:j + 1] = list(reversed(new_sol[v][i:j + 1]))
            move_key      = ('2opt', v, i, j)
            neighbors.append((new_sol, move_key))

        return neighbors

    def _get_neighbors(self, solution: Solution) -> List[Tuple[Solution, tuple]]:
        """
        Tạo danh sách láng giềng từ cả 3 operator, sắp xếp theo chi phí tăng dần.
        """
        neighbors  = []
        neighbors += self._move_relocate(solution)
        neighbors += self._move_swap(solution)
        neighbors += self._move_2opt_intra(solution)
        neighbors.sort(key=lambda x: self._total_dist(x[0]))
        return neighbors

    # ──────────────────────────────────────────────────────────────────
    # Main solve
    # ──────────────────────────────────────────────────────────────────

    def solve(self, initial_state: Solution) -> Tuple[Solution, float]:
        """
        Chạy Tabu Search từ initial_state.
        Dừng khi đạt max_iter hoặc không cải thiện sau max_no_improve vòng.
        Trả về (best_solution, best_distance_meters).
        """
        best_state = self._copy_solution(initial_state)
        best_dist  = self._total_dist(best_state)
        curr_state = self._copy_solution(initial_state)

        no_improve_count = 0

        for iteration in range(self.max_iter):

            if no_improve_count >= self.max_no_improve:
                print(f"[Tabu] Dừng sớm tại vòng {iteration}: "
                      f"{no_improve_count} vòng không cải thiện "
                      f"(ngưỡng={self.max_no_improve}).")
                break

            neighbors = self._get_neighbors(curr_state)
            if not neighbors:
                no_improve_count += 1
                continue

            best_non_tabu        = None
            best_non_tabu_dist   = float('inf')
            best_aspiration      = None
            best_aspiration_dist = float('inf')

            for next_state, move_key in neighbors:
                next_dist = self._total_dist(next_state)
                in_tabu   = move_key in self.tabu_list

                if not in_tabu:
                    if next_dist < best_non_tabu_dist:
                        best_non_tabu_dist = next_dist
                        best_non_tabu      = (next_state, move_key)
                else:
                    # Aspiration criterion: chấp nhận nếu tốt hơn best toàn cục
                    if (next_dist < best_dist
                            and next_dist < best_aspiration_dist):
                        best_aspiration_dist = next_dist
                        best_aspiration      = (next_state, move_key)

            # Ưu tiên aspiration nếu tốt hơn non-tabu tốt nhất
            chosen = None
            if best_aspiration and best_aspiration_dist < best_non_tabu_dist:
                chosen = best_aspiration
            elif best_non_tabu:
                chosen = best_non_tabu
            elif best_aspiration:
                chosen = best_aspiration

            if chosen is None:
                no_improve_count += 1
                continue

            next_state, move_key = chosen
            curr_state = next_state

            curr_dist = self._total_dist(curr_state)
            if curr_dist < best_dist:
                best_state       = self._copy_solution(curr_state)
                best_dist        = curr_dist
                no_improve_count = 0
            else:
                no_improve_count += 1

            self.tabu_list.append(move_key)

            if iteration % 100 == 0:
                print(f"  Iter {iteration:6d} | "
                      f"Curr={curr_dist:10.0f}m ({curr_dist/1000:.2f}km) | "
                      f"Best={best_dist:10.0f}m ({best_dist/1000:.2f}km) | "
                      f"NoImprove={no_improve_count}/{self.max_no_improve}")

        print(f"\n[Tabu Search] Hoàn tất. Best = {best_dist:.0f}m "
              f"({best_dist / 1000:.2f} km)")
        return best_state, best_dist