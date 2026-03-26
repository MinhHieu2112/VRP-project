import numpy as np
import copy
import time

class TabuSearchSolver:
    def __init__(self, distance_matrix, capacity, max_v, tabu_size=30, max_iter=1000, max_runtime=180):
        self.matrix = distance_matrix
        self.capacity = capacity
        self.max_v = max_v
        self.tabu_size = tabu_size
        self.max_iter = max_iter
        self.max_runtime = max_runtime
        self.tabu_list = []

    def _calculate_dist(self, state):
        total = 0
        for route in state:
            if len(route) > 2:
                for i in range(len(route) - 1):
                    total += self.matrix[route[i], route[i+1]]
        return total

    def _get_neighbors(self, state):
        neighbors = []
        for _ in range(40): 
            new_state = copy.deepcopy(state)
            v1, v2 = np.random.choice(len(new_state), 2, replace=False)
            if len(new_state[v1]) > 2 and len(new_state[v2]) > 2:
                p1 = np.random.randint(1, len(new_state[v1]) - 1)
                p2 = np.random.randint(1, len(new_state[v2]) - 1)
                new_state[v1][p1], new_state[v2][p2] = new_state[v2][p2], new_state[v1][p1]
                if (len(new_state[v1])-2 <= self.capacity) and (len(new_state[v2])-2 <= self.capacity):
                    neighbors.append((new_state, (v1, v2, p1, p2)))
        return neighbors

    def solve(self, initial_state):
        best_state = copy.deepcopy(initial_state)
        best_dist = self._calculate_dist(best_state)
        curr_state = copy.deepcopy(initial_state)
        
        start_time = time.time()

        for i in range(self.max_iter):
            # KIỂM TRA THỜI GIAN CHẠY (180 giây)
            if time.time() - start_time > self.max_runtime:
                print(f"--- Đã đạt giới hạn thời gian {self.max_runtime}s. Đang dừng... ---")
                break

            neighbors = self._get_neighbors(curr_state)
            if not neighbors: continue
            neighbors.sort(key=lambda x: self._calculate_dist(x[0]))
            
            for next_state, move in neighbors:
                next_dist = self._calculate_dist(next_state)
                if move not in self.tabu_list or next_dist < best_dist:
                    curr_state = next_state
                    if next_dist < best_dist:
                        best_state = copy.deepcopy(next_state)
                        best_dist = next_dist
                    self.tabu_list.append(move)
                    if len(self.tabu_list) > self.tabu_size: self.tabu_list.pop(0)
                    break
                    
            if i % 50 == 0:
                print(f"Vòng lặp {i} | Thời gian: {time.time()-start_time:.1f}s | Best: {best_dist:.2f} km")
                
        return best_state, best_dist