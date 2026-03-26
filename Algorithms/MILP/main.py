import json
import pandas as pd
import numpy as np
from pulp import *
import time

def solve_acvrp(matrix_path, config_path, limit_nodes=20):
    print("--- BẮT ĐẦU ĐỌC DỮ LIỆU ---")
    
    # 1. Đọc file config.json
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {config_path}")
        return

    # 2. Đọc và xử lý ma trận khoảng cách
    df = pd.read_csv(matrix_path, header=None)
    matrix_full = df.values
    
    # Tự động định dạng lại nếu ma trận bị dính thành 1 hàng
    if matrix_full.shape[0] == 1:
        n_total = int(np.sqrt(matrix_full.size))
        matrix_full = matrix_full.reshape((n_total, n_total))
        print(f"[*] Đã định dạng lại ma trận: {n_total}x{n_total}")

    # 3. Giới hạn số lượng điểm để MILP có thể giải được
    # (Với 1600 điểm, MILP sẽ không bao giờ kết thúc trên PC)
    n = min(limit_nodes, len(matrix_full))
    matrix = matrix_full[:n, :n]
    
    nodes = list(range(n))
    customers = list(range(1, n)) # Điểm 0 là Depot
    
    Q = config.get('vehicle_capacity', 100)
    K = config.get('num_vehicles', 10)
    raw_demands = config.get('demands', 1)

    # 4. Xử lý logic Demand (Sửa lỗi TypeError: 'int' object is not subscriptable)
    # Nếu demands là int, tạo dictionary gán cho mọi khách hàng giá trị đó
    if isinstance(raw_demands, int):
        demands = {i: raw_demands for i in customers}
        demands[0] = 0 # Depot không có nhu cầu
    else:
        # Nếu là list, lấy n phần tử đầu tiên
        demands = {i: raw_demands[i] for i in nodes}

    print(f"[*] Đang giải bài toán quy mô: {n} điểm (1 Depot + {n-1} khách hàng)")
    print(f"[*] Thông số: {K} xe | Sức chứa: {Q} | Nhu cầu mặc định: {raw_demands}")

    # --- MÔ HÌNH HÓA VỚI PuLP ---
    prob = LpProblem("ACVRP_Optimization", LpMinimize)

    # Biến quyết định: x[i][j] = 1 nếu xe đi từ i đến j
    x = LpVariable.dicts("x", (nodes, nodes), 0, 1, cat=LpBinary)
    
    # Biến luồng: f[i][j] là số lượng hàng trên xe khi đi qua cạnh (i, j)
    # Đây là phương pháp Single Commodity Flow để khử subtour và quản lý tải trọng
    f = LpVariable.dicts("f", (nodes, nodes), 0, Q, cat=LpContinuous)

    # Hàm mục tiêu: Tối thiểu hóa tổng khoảng cách
    prob += lpSum([matrix[i][j] * x[i][j] for i in nodes for j in nodes if i != j])

    # RÀNG BUỘC:
    # 1. Mỗi khách hàng phải có đúng 1 xe đến và 1 xe đi
    for i in customers:
        prob += lpSum([x[j][i] for j in nodes if i != j]) == 1
        prob += lpSum([x[i][j] for j in nodes if i != j]) == 1

    # 2. Giới hạn số xe rời kho Depot
    prob += lpSum([x[0][j] for j in customers]) <= K

    # 3. Bảo toàn luồng (Flow Conservation): Khử chu trình con & trừ dần nhu cầu
    # Lượng hàng vào node i - Lượng hàng ra khỏi node i = Nhu cầu của i
    for i in customers:
        prob += lpSum([f[j][i] for j in nodes if i != j]) - \
                lpSum([f[i][j] for j in nodes if i != j]) == demands[i]

    # 4. Liên kết giữa biến x và biến f (Lượng hàng chỉ tồn tại trên cạnh được chọn)
    for i in nodes:
        for j in nodes:
            if i != j:
                # Lượng hàng trên xe không được vượt quá tải trọng Q khi đi qua cạnh (i, j)
                prob += f[i][j] <= Q * x[i][j]

    # --- GIẢI BÀI TOÁN ---
    timelimit = config.get('max_runtime_seconds', 120)
    print(f"\n--- ĐANG GIẢI (Giới hạn: {timelimit}s) ---")
    
    # msg=1 để xem log của Solver, giúp bạn biết nó có đang chạy hay không
    status = prob.solve(PULP_CBC_CMD(timeLimit=timelimit, msg=1))

    # --- KẾT QUẢ ---
    print("\n" + "="*50)
    print(f"TRẠNG THÁI: {LpStatus[status]}")
    
    if LpStatus[status] in ['Optimal', 'Feasible']:
        print(f"TỔNG KHOẢNG CÁCH: {value(prob.objective):.2f}")
        
        # Trích xuất lộ trình
        print("\nCHI TIẾT LỘ TRÌNH CÁC XE:")
        for j in customers:
            if value(x[0][j]) is not None and value(x[0][j]) > 0.5:
                route = [0, j]
                curr = j
                current_route_load = demands[j]
                
                # Tìm các node tiếp theo cho đến khi về lại kho (0)
                while curr != 0:
                    for next_node in nodes:
                        if curr != next_node and value(x[curr][next_node]) is not None and value(x[curr][next_node]) > 0.5:
                            route.append(next_node)
                            if next_node != 0:
                                current_route_load += demands[next_node]
                            curr = next_node
                            break
                    if len(route) > n: break # Tránh vòng lặp vô tận nếu có lỗi
                
                print(f" > Xe: {' -> '.join(map(str, route))} | Tải trọng: {current_route_load}/{Q}")
    else:
        print("Solver không tìm thấy lời giải trong thời gian cho phép.")
    print("="*50)

if __name__ == "__main__":
    # Lưu ý: Tôi để mặc định giải 15 điểm để bạn chạy thử lấy kết quả ngay.
    # Muốn tăng lên, hãy sửa tham số limit_nodes (nhưng MILP sẽ rất chậm sau 30-40 điểm).
    solve_acvrp('orsm_matrix_scaled.csv', 'config.json', limit_nodes=50)