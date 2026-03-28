import json
import pandas as pd
import numpy as np
from milp_solvers import solve_acvrp_milp # Import hàm từ file thuật toán

def load_and_prep_data(matrix_path, config_path, limit_nodes):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {config_path}")
        return None, None, None

    df = pd.read_csv(matrix_path, header=None)
    matrix_full = df.values
    
    if matrix_full.shape[0] == 1:
        n_total = int(np.sqrt(matrix_full.size))
        matrix_full = matrix_full.reshape((n_total, n_total))

    # KHẮC PHỤC 4: Cắt ma trận (Cảnh báo người dùng)
    n = min(limit_nodes, len(matrix_full))
    print(f"[*] Đang sử dụng {n} điểm đầu tiên trong ma trận dữ liệu làm Input.")
    matrix = matrix_full[:n, :n]
    
    nodes = list(range(n))
    customers = list(range(1, n))
    
    raw_demands = config.get('demands', 1)

    # KHẮC PHỤC 3: Xử lý demands an toàn
    demands = {0: 0} # Depot mặc định bằng 0
    if isinstance(raw_demands, int):
        for i in customers:
            demands[i] = raw_demands
    elif isinstance(raw_demands, list):
        if len(raw_demands) < n:
            raise ValueError(f"Danh sách demands ({len(raw_demands)}) ngắn hơn số điểm đang giải ({n}).")
        for i in customers:
            demands[i] = raw_demands[i]
    else:
        raise TypeError("Demands trong config phải là số nguyên hoặc mảng/list.")

    return matrix, demands, config

if __name__ == "__main__":
    matrix_path = 'orsm_matrix_scaled.csv'
    config_path = 'config.json'
    limit_nodes = 350 # Giới hạn 25 điểm để test MILP
    
    print("--- BẮT ĐẦU ĐỌC DỮ LIỆU ---")
    matrix, demands, config = load_and_prep_data(matrix_path, config_path, limit_nodes)
    
    if matrix is not None:
        Q = config.get('vehicle_capacity', 10)
        K = config.get('num_vehicles', 10)
        timelimit = config.get('max_runtime_seconds', 120)

        print(f"[*] Quy mô: {len(matrix)} điểm | {K} xe | Sức chứa: {Q}")
        print(f"--- ĐANG GIẢI BẰNG MILP (Giới hạn: {timelimit}s) ---")
        
        # Gọi thuật toán từ file milp_solver.py
        status_str, obj_val, routes_info = solve_acvrp_milp(
            matrix, demands, num_vehicles=K, capacity=Q, timelimit=timelimit
        )

        print("\n" + "="*50)
        print(f"TRẠNG THÁI: {status_str}")
        
        if obj_val is not None:
            print(f"TỔNG KHOẢNG CÁCH: {obj_val/100:.2f}")
            print("\nCHI TIẾT LỘ TRÌNH CÁC XE:")
            for info in routes_info:
                r_str = ' -> '.join(map(str, info['route']))
                valid_str = "HỢP LỆ" if info['is_valid'] else "QUÁ TẢI"
                print(f" > Xe: {r_str} | Tải trọng: {info['load']}/{Q} [{valid_str}]")
        else:
            print("Solver không tìm thấy nghiệm nguyên khả thi nào.")
        print("="*50)