import numpy as np
import copy
import time
from collections import deque  # FIX #3: dùng deque thay list cho O(1) pop


class TabuSearchSolver:
    def __init__(self, distance_matrix, demands, capacity, max_v,
                 tabu_size=30, max_iter=10000, max_runtime=180):
        self.matrix = distance_matrix
        self.demands = demands
        self.capacity = capacity
        self.max_v = max_v
        self.tabu_size = tabu_size
        self.max_iter = max_iter
        self.max_runtime = max_runtime
        # FIX #3: deque với maxlen tự động loại phần tử cũ nhất → O(1)
        self.tabu_list = deque(maxlen=tabu_size)

    def _route_dist(self, route):
        total = 0.0
        for i in range(len(route) - 1):
            total += self.matrix[route[i], route[i + 1]]
        return total

    def _total_dist(self, state):
        return sum(self._route_dist(r) for r in state if len(r) > 2)

    def _route_demand(self, route):
        return sum(self.demands.get(node, 0) for node in route if node != 0)

    def _is_feasible(self, route):
        return self._route_demand(route) <= self.capacity

    def _move_relocate(self, state):
        neighbors = []
        n_routes = len(state)

        for _ in range(30):
            v1 = np.random.randint(0, n_routes)
            v2 = np.random.randint(0, n_routes)
            if v1 == v2 or len(state[v1]) <= 3:
                continue

            p1 = np.random.randint(1, len(state[v1]) - 1)
            customer = state[v1][p1]

            for p2 in range(1, len(state[v2])):
                new_state = copy.deepcopy(state)
                new_state[v1].pop(p1)
                new_state[v2].insert(p2, customer)

                if self._is_feasible(new_state[v2]):
                    move_key = ('relocate', v1, v2, customer, p2)
                    neighbors.append((new_state, move_key))
                    break

        return neighbors

    def _move_swap(self, state):
        neighbors = []

        for _ in range(20):
            v1, v2 = np.random.choice(len(state), 2, replace=False)
            if len(state[v1]) <= 2 or len(state[v2]) <= 2:
                continue

            p1 = np.random.randint(1, len(state[v1]) - 1)
            p2 = np.random.randint(1, len(state[v2]) - 1)

            new_state = copy.deepcopy(state)
            new_state[v1][p1], new_state[v2][p2] = new_state[v2][p2], new_state[v1][p1]

            if self._is_feasible(new_state[v1]) and self._is_feasible(new_state[v2]):
                c1 = state[v1][p1]
                c2 = state[v2][p2]
                move_key = ('swap', min(c1, c2), max(c1, c2))
                neighbors.append((new_state, move_key))

        return neighbors

    def _move_2opt_intra(self, state):
        neighbors = []

        for _ in range(20):
            v = np.random.randint(0, len(state))
            route = state[v]
            if len(route) < 5:
                continue

            i = np.random.randint(1, len(route) - 2)
            j = np.random.randint(i + 1, len(route) - 1)

            new_state = copy.deepcopy(state)
            new_state[v][i:j + 1] = new_state[v][i:j + 1][::-1]
            move_key = ('2opt', v, i, j)
            neighbors.append((new_state, move_key))

        return neighbors

    def _get_neighbors(self, state):
        neighbors = []
        neighbors += self._move_relocate(state)
        neighbors += self._move_swap(state)
        neighbors += self._move_2opt_intra(state)
        neighbors.sort(key=lambda x: self._total_dist(x[0]))
        return neighbors

    def solve(self, initial_state):
        best_state = copy.deepcopy(initial_state)
        best_dist = self._total_dist(best_state)
        curr_state = copy.deepcopy(initial_state)

        start_time = time.time()

        for iteration in range(self.max_iter):
            elapsed = time.time() - start_time
            if elapsed > self.max_runtime:
                print(f"--- Đã đạt giới hạn thời gian {self.max_runtime}s. "
                      f"Dừng tại vòng {iteration}. ---")
                break

            neighbors = self._get_neighbors(curr_state)
            if not neighbors:
                continue

            # FIX #4: Tách biệt rõ ràng aspiration criterion
            # Bước 1: tìm neighbor tốt nhất không bị tabu
            # Bước 2: nếu không có, chấp nhận neighbor bị tabu
            #         nếu nó tốt hơn best_dist (aspiration)
            best_non_tabu = None
            best_non_tabu_dist = float('inf')
            best_aspiration = None
            best_aspiration_dist = float('inf')

            for next_state, move_key in neighbors:
                next_dist = self._total_dist(next_state)
                in_tabu = move_key in self.tabu_list

                if not in_tabu:
                    if next_dist < best_non_tabu_dist:
                        best_non_tabu_dist = next_dist
                        best_non_tabu = (next_state, move_key)
                else:
                    # Aspiration criterion: chấp nhận nếu vượt best
                    if next_dist < best_dist and next_dist < best_aspiration_dist:
                        best_aspiration_dist = next_dist
                        best_aspiration = (next_state, move_key)

            # Chọn move: ưu tiên aspiration nếu tốt hơn non-tabu best
            chosen = None
            if best_aspiration and best_aspiration_dist < best_non_tabu_dist:
                chosen = best_aspiration
            elif best_non_tabu:
                chosen = best_non_tabu
            elif best_aspiration:
                chosen = best_aspiration

            if chosen is None:
                continue

            next_state, move_key = chosen
            next_dist = self._total_dist(next_state)
            curr_state = next_state

            if next_dist < best_dist:
                best_state = copy.deepcopy(next_state)
                best_dist = next_dist

            # FIX #3: deque.append() tự động pop phần tử cũ khi đầy
            self.tabu_list.append(move_key)

            if iteration % 100 == 0:
                print(f"  Iter {iteration:5d} | Time: {elapsed:6.1f}s | "
                      f"Curr: {self._total_dist(curr_state):8.2f} | "
                      f"Best: {best_dist:8.2f}")

        print(f"\n[Tabu Search] Hoàn tất. Best distance = {best_dist:.2f}")
        return best_state, best_dist