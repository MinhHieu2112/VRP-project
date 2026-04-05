import numpy as np
import time
import json
import os
import sys
import math
import threading
from alns.stop import MaxIterations


class NoImprovementStop:
    """
    Dừng khi không cải thiện best solution sau `max_no_improve` vòng liên tiếp.
    """
    def __init__(self, max_no_improve: int):
        self._max_no_improve = max_no_improve
        self._no_improve_count = 0
        self._best_obj = float('inf')

    def __call__(self, rng, best, curr):
        obj = best.objective()
        if obj < self._best_obj - 1e-6:
            self._best_obj = obj
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1
        return self._no_improve_count >= self._max_no_improve


# Import local modules
from src.utils.loader import load_distance_matrix
from src.state import CvrpState
from src.solver import configure_alns

# Import Project Utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'  , '..'))
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer


def load_config(path='config.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_initial_solution(clients, cap, matrix, demands, max_v):
    """
    Tạo nghiệm ban đầu bằng cách nhóm khách theo capacity thực tế,
    không chia đều theo số xe. Mỗi route được lấp đầy tối đa cap khách
    trước khi mở route mới — tránh tình trạng 1 khách/xe.

    Nếu tổng số route vượt max_v thì gom thêm vào route cuối
    (ALNS sẽ xử lý vi phạm capacity sau).
    """
    routes = []
    current_route = [0]
    current_load = 0

    for client in clients:
        d = demands[client]
        if current_load + d > cap:
            current_route.append(0)
            routes.append(current_route)
            current_route = [0]
            current_load = 0
        current_route.append(client)
        current_load += d

    if len(current_route) > 1:
        current_route.append(0)
        routes.append(current_route)

    # Nếu vượt max_v, gom các route cuối vào route trước
    while len(routes) > max_v:
        last = routes.pop()
        routes[-1] = routes[-1][:-1] + last[1:]  # bỏ depot cuối rồi nối

    return routes


def main():
    config = load_config()
    matrix = load_distance_matrix(config['data_path'])
    if matrix is None:
        return

    num_nodes = matrix.shape[0]
    scaling = config.get('common_model_parameters', {}).get('scaling_factor', 1.0)
    constraints = config['constraints']
    max_v = constraints['max_vehicles']
    cap   = constraints['vehicle_capacity']

    # Khởi tạo demands
    demands = np.ones(num_nodes) * constraints.get('default_demand', 1)
    demands[0] = 0

    clients = list(range(1, num_nodes))

    # ── Tạo nghiệm ban đầu dựa trên capacity, không chia đều theo xe ──
    initial_routes = build_initial_solution(clients, cap, matrix, demands, max_v)
    initial_state  = CvrpState(initial_routes, [], matrix, cap, demands, config)

    num_init_vehicles = len(initial_routes)
    init_dist = sum(initial_state.route_cost(r) for r in initial_routes) / scaling
    print(f"[*] {num_nodes-1} khách hàng | {num_init_vehicles} xe ban đầu | capacity={cap}")
    print(f"Quãng đường ban đầu: {init_dist:.2f} km")

    # 1. Cấu hình ALNS
    alns, accept, select, _ = configure_alns(initial_state, config)

    # 2. Stopping criterion
    p = config['alns_parameters']
    max_no_improve = p.get('max_no_improve', 3000)
    print(f"--- Bắt đầu tối ưu (dừng sau {max_no_improve} vòng không cải thiện) ---")

    # --- Progress logging ---
    best_so_far      = [init_dist]
    best_unassigned  = [0]
    improvement_count = [0]

    def on_best(state, rnd):
        dist = sum(state.route_cost(r) for r in state.routes if len(r) > 2) / scaling
        best_so_far[0]       = dist
        best_unassigned[0]   = len(state.unassigned)
        improvement_count[0] += 1
        sys.stdout.write(
            f"\n  -> [#{improvement_count[0]}] Cải thiện: {dist:.2f} km"
            f" | Unassigned: {len(state.unassigned)}\n"
        )
        sys.stdout.flush()

    alns.on_best(on_best)

    # Thread in tiến độ mỗi 10 giây
    stop_flag  = threading.Event()
    start_time = time.time()

    def progress_printer():
        while not stop_flag.is_set():
            stop_flag.wait(10)
            if stop_flag.is_set():
                break
            elapsed = time.time() - start_time
            sys.stdout.write(
                f"\r  [{elapsed:5.0f}s] Đang tối ưu... "
                f"Best: {best_so_far[0]:.2f} km | "
                f"Unassigned: {best_unassigned[0]} | "
                f"Cải thiện: {improvement_count[0]} lần   "
            )
            sys.stdout.flush()

    t = threading.Thread(target=progress_printer, daemon=True)
    t.start()

    stop   = NoImprovementStop(max_no_improve=max_no_improve)
    result = alns.iterate(initial_state, select, accept, stop=stop)

    stop_flag.set()
    t.join(timeout=1)

    best_state = result.best_state

    # 3. Local Search 2-opt
    print("\n--- Đang làm mịn lộ trình với 2-opt ---")
    best_state.apply_2opt()

    end_time = time.time()

    # 4. Xuất kết quả
    actual_routes = [r for r in best_state.routes if len(r) > 2]
    routes_dict   = {i: [int(n) for n in r] for i, r in enumerate(actual_routes)}
    final_dist    = sum(best_state.route_cost(r) for r in actual_routes) / scaling

    # Thống kê phân bố tải trọng
    loads = []
    for r in actual_routes:
        load = sum(demands[n] for n in r if n != 0)
        loads.append(int(load))
    avg_load = np.mean(loads) if loads else 0
    avg_pts  = np.mean([len(r)-2 for r in actual_routes]) if actual_routes else 0

    standardized_result = {
        "solver_name":        "ALNS_Full_Optimized",
        "total_distance_km":  final_dist,
        "execution_time":     end_time - start_time,
        "routes":             routes_dict,
        "num_vehicles":       len(routes_dict),
    }

    output_dir = os.path.join(os.path.dirname(__file__), '..'  , '..', 'Results', 'ALNS')
    os.makedirs(output_dir, exist_ok=True)
    ResultHandler.save_to_txt(standardized_result, output_dir)

    print(f"\n[HOÀN TẤT]")
    print(f"Tổng quãng đường: {final_dist:.2f} km")
    print(f"Số xe sử dụng:    {len(routes_dict)}")
    print(f"Trung bình:       {avg_pts:.1f} điểm/xe | {avg_load:.1f} tải/xe (capacity={cap})")
    print(f"Thời gian:        {end_time - start_time:.2f} giây")
    print(f"Số lần cải thiện: {improvement_count[0]}")


if __name__ == "__main__":
    main()