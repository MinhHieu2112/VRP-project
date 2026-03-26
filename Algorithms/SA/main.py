import os
import json
import time
import pandas as pd

from solver_sa import SimulatedAnnealingSolver

# ===== XÁC ĐỊNH CÁC ĐƯỜNG DẪN GỐC =====
# File này đang nằm ở: VRP-project/Algorithms/SA/main.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # Thư mục SA
# Nhảy lên 2 cấp để ra VRP-project
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR)) 

CONFIG_PATH = os.path.join(CURRENT_DIR, "configSA.json")

# ===== LOAD CONFIG =====
def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Không tìm thấy file config tại: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ===== LOAD DATA =====
def load_data(config):
    # Sử dụng project_root làm mốc để tìm thư mục Data
    # Nếu config['data_path'] là "../../Data/osrm_matrix.csv" 
    # thì os.path.join(CURRENT_DIR, ...) sẽ tự xử lý các dấu ".." chính xác.
    
    matrix_path = os.path.normpath(os.path.join(CURRENT_DIR, config['data_path']))
    locations_path = os.path.normpath(os.path.join(CURRENT_DIR, config['locations_path']))

    print("--- Kiểm tra đường dẫn ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Matrix Path:  {matrix_path}")
    print(f"Locs Path:    {locations_path}")
    print("--------------------------")

    if not os.path.exists(matrix_path):
        # Nếu vẫn lỗi, thử tìm phương án dự phòng trực tiếp từ Project Root
        matrix_path = os.path.join(PROJECT_ROOT, "Data", "osrm_matrix.csv")
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"LỖI: Không tìm thấy file matrix tại bất kỳ đâu!\nĐã thử: {matrix_path}")

    if not os.path.exists(locations_path):
        locations_path = os.path.join(PROJECT_ROOT, "Data", "locations.csv")
        if not os.path.exists(locations_path):
            raise FileNotFoundError(f"LỖI: Không tìm thấy file locations!")

    # Đọc dữ liệu
    dist = pd.read_csv(matrix_path, header=None).values
    df_locations = pd.read_csv(locations_path)

    return dist, df_locations

# ===== RUN SA =====
def run_sa():
    print("\n===== RUN SIMULATED ANNEALING (SA) =====")

    config = load_config()
    dist, df_locations = load_data(config)

    data_bundle = {
        "distance_matrix": dist,
        "df_locations": df_locations
    }

    solver = SimulatedAnnealingSolver(data_bundle, config)

    # ===== SOLVE =====
    start = time.time()
    routes, total_cost = solver.solve()
    runtime = time.time() - start

    # Kiểm tra key trong config để tránh lỗi KeyError
    common_params = config.get('common_model_parameters', {})
    scaling = common_params.get('scaling_factor', 1.0)

    result = {
        "solver_name": "Simulated Annealing",
        "total_distance_km": total_cost / scaling,
        "execution_time": runtime,
        "routes": routes,
        "num_vehicles": len(routes)
    }

    # ===== PRINT =====
    print("\n===== RESULT =====")
    print(f"Số xe: {result['num_vehicles']}")
    print(f"Tổng quãng đường: {result['total_distance_km']:.2f} km")
    print(f"Thời gian chạy: {result['execution_time']:.2f} s")

    # ===== SAVE RESULT =====
    # Lưu vào thư mục Results nằm cùng cấp với main.py
    out_dir = os.path.join(CURRENT_DIR, "Results")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    res_path = os.path.join(out_dir, "sa_result.txt")

    with open(res_path, "w", encoding="utf-8") as f:
        f.write(f"Tổng quãng đường thực tế: {result['total_distance_km']:.2f} km\n")
        f.write(f"Số xe sử dụng: {result['num_vehicles']}\n")
        f.write(f"Giá trị Objective cuối cùng: {total_cost:.2f}\n")
        f.write(f"Tổng thời gian chạy: {runtime:.2f} giây\n")
        f.write("-" * 40 + "\n")

        for idx, (k, route) in enumerate(routes.items()):
            if len(route) > 2:
                f.write(f"Route #{idx+1} (Xe {k}): {' '.join(map(str, route))}\n")

    print(f"\nKết quả đã ghi vào file: {res_path}")
    return result

if __name__ == "__main__":
    run_sa()