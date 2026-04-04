import random
import math
import time

class SimulatedAnnealingSolver:
    def __init__(self, data_bundle, config):
        self.dist = data_bundle['distance_matrix']
        self.n = len(self.dist)

        cons = config.get('constraints', {})
        self.capacity = cons.get('vehicle_capacity', 10)
        self.demand   = cons.get('default_demand', 1)

        sa_cfg = config.get('alns_parameters', {})
        self.max_runtime = sa_cfg.get('max_runtime', 180)
        self.T_start     = sa_cfg.get('start_temperature', 5000)
        self.T_min       = sa_cfg.get('end_temperature', 0.1)
        self.alpha        = sa_cfg.get('step', 0.9995)

        # Với 1600 nodes ưu tiên distance: ít iter/T nhưng nhiều bước hạ nhiệt
        self.iter_per_T = 100

        self.all_customers = set(range(1, self.n))

        # Penalty xe nhỏ thôi vì ưu tiên distance
        # (160 xe đã là tối thiểu → không cần ép giảm thêm)
        self.vehicle_penalty = 500

    # ──────────────────────────────────────────────────────────────────
    #  KHỞI TẠO: Greedy Nearest-Neighbor → nghiệm ban đầu tốt hơn ~30%
    # ──────────────────────────────────────────────────────────────────
    def initial_solution(self):
        """
        Nearest-neighbor greedy: mỗi xe chọn khách hàng gần nhất hiện tại.
        Giảm quãng đường ban đầu so với random shuffle rất nhiều.
        """
        unvisited = list(range(1, self.n))
        # Sắp xếp theo góc từ depot để các xe không chồng lấp vùng phục vụ
        # (heuristic đơn giản: cluster theo vị trí trong dist matrix)
        random.shuffle(unvisited)  # vẫn giữ 1 chút ngẫu nhiên để SA đa dạng

        routes = []
        while unvisited:
            route   = [0]
            load    = 0
            current = 0

            while load + self.demand <= self.capacity and unvisited:
                # Nearest neighbor từ node hiện tại
                best_dist = float('inf')
                best_node = None
                best_idx  = -1
                for idx, node in enumerate(unvisited):
                    d = self.dist[current][node]
                    if d < best_dist:
                        best_dist = d
                        best_node = node
                        best_idx  = idx

                route.append(best_node)
                unvisited.pop(best_idx)
                load    += self.demand
                current  = best_node

            route.append(0)
            routes.append(route)

        return routes

    # ──────────────────────────────────────────────────────────────────
    #  UTILITY
    # ──────────────────────────────────────────────────────────────────
    def route_cost(self, route):
        if len(route) < 2:
            return 0
        return sum(self.dist[route[i]][route[i+1]]
                   for i in range(len(route) - 1))

    def route_load(self, route):
        return sum(1 for node in route if node != 0) * self.demand

    def total_cost(self, routes):
        active   = [r for r in routes if len(r) > 2]
        distance = sum(self.route_cost(r) for r in active)
        return distance + len(active) * self.vehicle_penalty

    def _is_feasible(self, routes):
        visited = []
        for r in routes:
            if r[0] != 0 or r[-1] != 0:
                return False
            customers = [node for node in r if node != 0]
            if len(customers) * self.demand > self.capacity:
                return False
            visited.extend(customers)
        return sorted(visited) == sorted(self.all_customers)

    # ──────────────────────────────────────────────────────────────────
    #  TOÁN TỬ 1: 2-OPT CÓ HƯỚNG (trong 1 route)
    #  → Tìm cặp (i,j) thực sự cải thiện, không random mù
    # ──────────────────────────────────────────────────────────────────
    def _two_opt_improving(self, route):
        """
        Thực hiện 2-opt và chỉ chấp nhận nếu delta < 0 (luôn cải thiện).
        Trả về route mới nếu tìm được cải thiện, None nếu không.
        """
        best_delta = 0
        best_i, best_j = -1, -1
        n = len(route)

        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Chi phí hiện tại: cạnh (i-1,i) + (j, j+1)
                # Sau đảo:           cạnh (i-1,j) + (i, j+1)
                delta = (
                    self.dist[route[i-1]][route[j]]
                    + self.dist[route[i]][route[j+1]]
                    - self.dist[route[i-1]][route[i]]
                    - self.dist[route[j]][route[j+1]]
                )
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j

        if best_i >= 0:
            new_r = route[:]
            new_r[best_i:best_j+1] = new_r[best_i:best_j+1][::-1]
            return new_r
        return None

    # ──────────────────────────────────────────────────────────────────
    #  TOÁN TỬ 2: OR-OPT (di chuyển chuỗi 1-2-3 nodes sang route khác)
    #  → Toán tử hiệu quả nhất để giảm quãng đường VRP
    # ──────────────────────────────────────────────────────────────────
    def _or_opt(self, routes, seg_len=1):
        """
        Lấy đoạn seg_len nodes từ route nguồn,
        tìm vị trí cheapest insertion trong route đích.
        Chỉ thực hiện nếu delta < 0.
        """
        active_idx = [i for i, r in enumerate(routes) if len(r) > 2]
        if len(active_idx) < 2:
            return routes

        random.shuffle(active_idx)

        for src_idx in active_idx:
            src = routes[src_idx]
            customers_in_src = len(src) - 2
            if customers_in_src < seg_len + 1:
                continue  # Không lấy hết toàn bộ xe

            # Thử từng vị trí trong src
            for start in range(1, len(src) - 1 - seg_len + 1):
                seg = src[start:start + seg_len]

                # Chi phí tháo đoạn seg ra khỏi src
                before   = src[start - 1]
                after    = src[start + seg_len]
                cost_remove = (
                    self.dist[before][after]
                    - self.dist[before][seg[0]]
                    - self.dist[seg[-1]][after]
                )  # âm = tiết kiệm được khi tháo ra

                # Tìm route đích tốt nhất để chèn seg vào
                best_gain  = 0  # chỉ chấp nhận nếu tổng gain > 0
                best_dst   = -1
                best_pos   = -1

                for dst_idx in active_idx:
                    if dst_idx == src_idx:
                        continue
                    dst = routes[dst_idx]
                    # Kiểm tra capacity
                    if self.route_load(dst) + seg_len * self.demand > self.capacity:
                        continue

                    for pos in range(1, len(dst)):
                        cost_insert = (
                            self.dist[dst[pos-1]][seg[0]]
                            + self.dist[seg[-1]][dst[pos]]
                            - self.dist[dst[pos-1]][dst[pos]]
                        )
                        gain = -(cost_remove + cost_insert)  # gain dương = tốt hơn
                        if gain > best_gain:
                            best_gain = gain
                            best_dst  = dst_idx
                            best_pos  = pos

                # Nếu tìm được nước đi có lợi → thực hiện
                if best_dst >= 0:
                    new_routes = [r[:] for r in routes]
                    # Tháo seg khỏi src
                    del new_routes[src_idx][start:start + seg_len]
                    # Chèn vào dst (vị trí best_pos có thể lệch nếu src==dst nhưng đã loại trừ)
                    for k, node in enumerate(seg):
                        new_routes[best_dst].insert(best_pos + k, node)
                    return [r for r in new_routes if len(r) > 2]

        return routes  # Không tìm được cải thiện

    # ──────────────────────────────────────────────────────────────────
    #  TOÁN TỬ 3: 2-OPT* LIÊN TUYẾN (hoán đổi đuôi giữa 2 routes)
    #  → Tối ưu quãng đường liên xe, 2-opt thường không làm được
    # ──────────────────────────────────────────────────────────────────
    def _two_opt_star(self, routes):
        """
        Hoán đổi phần đuôi giữa 2 routes:
          r1: [0 ... A | B ... 0]
          r2: [0 ... C | D ... 0]
          →  r1: [0 ... A | D ... 0]
             r2: [0 ... C | B ... 0]
        Chỉ thực hiện nếu feasible và delta < 0.
        """
        active_idx = [i for i, r in enumerate(routes) if len(r) > 2]
        if len(active_idx) < 2:
            return routes

        r1_idx, r2_idx = random.sample(active_idx, 2)
        r1, r2 = routes[r1_idx], routes[r2_idx]

        best_delta = 0
        best_i, best_j = -1, -1

        for i in range(1, len(r1) - 1):
            # Số khách hàng sau điểm i trong r1
            tail1_len = len(r1) - 1 - i
            for j in range(1, len(r2) - 1):
                tail2_len = len(r2) - 1 - j

                # Kiểm tra capacity sau hoán đổi
                load_r1_new = (i - 1 + tail2_len) * self.demand  # phần đầu r1 + đuôi r2
                load_r2_new = (j - 1 + tail1_len) * self.demand
                if (load_r1_new > self.capacity or
                        load_r2_new > self.capacity):
                    continue

                # Delta cost
                delta = (
                    self.dist[r1[i-1]][r2[j]]
                    + self.dist[r2[j-1]][r1[i]]
                    - self.dist[r1[i-1]][r1[i]]
                    - self.dist[r2[j-1]][r2[j]]
                )
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j

        if best_i >= 0:
            new_routes = [r[:] for r in routes]
            # Hoán đổi đuôi
            tail1 = new_routes[r1_idx][best_i:]
            tail2 = new_routes[r2_idx][best_j:]
            new_routes[r1_idx] = new_routes[r1_idx][:best_i] + tail2
            new_routes[r2_idx] = new_routes[r2_idx][:best_j] + tail1
            return [r for r in new_routes if len(r) > 2]

        return routes

    # ──────────────────────────────────────────────────────────────────
    #  NEIGHBOR: Phân phối lại tỷ lệ toán tử cho bài toán distance
    # ──────────────────────────────────────────────────────────────────
    def get_neighbor(self, routes):
        p = random.random()

        # 35% Or-Opt seg=1 (relocate 1 node có hướng)
        if p < 0.35:
            return self._or_opt(routes, seg_len=1)

        # 20% Or-Opt seg=2 (di chuyển 2 nodes liên tiếp)
        elif p < 0.55:
            return self._or_opt(routes, seg_len=2)

        # 20% 2-opt* liên tuyến
        elif p < 0.75:
            return self._two_opt_star(routes)

        # 25% 2-opt có hướng trong 1 route
        else:
            active = [r for r in routes if len(r) > 2]
            if not active:
                return routes
            r_idx = random.randint(0, len(active) - 1)
            improved = self._two_opt_improving(active[r_idx])
            if improved:
                active[r_idx] = improved
            return active

    # ──────────────────────────────────────────────────────────────────
    #  SOLVE
    # ──────────────────────────────────────────────────────────────────
    def solve(self):
        start_time = time.time()

        current      = self.initial_solution()
        best         = [r[:] for r in current]
        current_cost = self.total_cost(current)
        best_cost    = current_cost

        init_dist = sum(self.route_cost(r) for r in current)
        print(f"[SA] Init: {len(current)} xe, dist={init_dist:.1f}, "
              f"T_start={self.T_start}, penalty/xe={self.vehicle_penalty}")

        T = self.T_start
        step = 0

        while T > self.T_min:
            if time.time() - start_time > self.max_runtime:
                break

            for _ in range(self.iter_per_T):
                neighbor      = self.get_neighbor(current)
                neighbor_cost = self.total_cost(neighbor)
                delta         = neighbor_cost - current_cost

                if delta < 0 or random.random() < math.exp(-min(delta / T, 700)):
                    current      = neighbor
                    current_cost = neighbor_cost

                    if current_cost < best_cost:
                        best      = [r[:] for r in current]
                        best_cost = current_cost

            T    *= self.alpha
            step += 1

            if step % 2000 == 0:
                n_v  = len([r for r in best if len(r) > 2])
                dist = sum(self.route_cost(r) for r in best)
                elapsed = time.time() - start_time
                print(f"  [step={step} T={T:.2f}] xe={n_v}, "
                      f"dist={dist:.1f}, time={elapsed:.1f}s")

        actual_dist = sum(self.route_cost(r) for r in best)
        routes_dict = {i: r for i, r in enumerate(best) if len(r) > 2}

        print(f"[SA] Done: {len(routes_dict)} xe, dist={actual_dist:.2f}")
        return routes_dict, actual_dist