import os
import sys
import json
import time
import pandas as pd

from solver_sa import SimulatedAnnealingSolver

# ===== XÁC ĐỊNH CÁC ĐƯỜNG DẪN GỐC =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# Import Utils chung từ project root
sys.path.append(PROJECT_ROOT)
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer

CONFIG_PATH = os.path.join(CURRENT_DIR, "configSA.json")

# ===== LOAD CONFIG =====
def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Không tìm thấy file config tại: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ===== LOAD DATA =====
def load_data(config):
    matrix_path = os.path.normpath(os.path.join(CURRENT_DIR, config['data_path']))
    locations_path = os.path.normpath(os.path.join(CURRENT_DIR, config['locations_path']))

    print("--- Kiểm tra đường dẫn ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Matrix Path:  {matrix_path}")
    print(f"Locs Path:    {locations_path}")
    print("--------------------------")

    if not os.path.exists(matrix_path):
        matrix_path = os.path.join(PROJECT_ROOT, "Data", "osrm_matrix.csv")
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"LỖI: Không tìm thấy file matrix!")

    if not os.path.exists(locations_path):
        locations_path = os.path.join(PROJECT_ROOT, "Data", "locations.csv")
        if not os.path.exists(locations_path):
            raise FileNotFoundError(f"LỖI: Không tìm thấy file locations!")

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

    # Kiểm tra scaling factor
    common_params = config.get('common_model_parameters', {})
    scaling = common_params.get('scaling_factor', 1.0)

    # === TẠO KẾT QUẢ CHUẨN ===
    # Lọc chỉ giữ route có khách hàng, đánh lại index từ 0
    clean_routes = {}
    idx = 0
    for k, route in routes.items():
        if len(route) > 2:
            clean_routes[idx] = route
            idx += 1

    standardized_result = {
        "solver_name": "SA",
        "total_distance_km": total_cost / scaling,
        "execution_time": runtime,
        "routes": clean_routes,
        "num_vehicles": len(clean_routes)
    }

    # ===== IN KẾT QUẢ =====
    print("\n===== RESULT =====")
    print(f"Số xe: {standardized_result['num_vehicles']}")
    print(f"Tổng quãng đường: {standardized_result['total_distance_km']:.2f} km")
    print(f"Thời gian chạy: {standardized_result['execution_time']:.2f} s")

    # ===== LƯU KẾT QUẢ BẰNG RESULTHANDLER CHUNG =====
    output_dir = os.path.join(PROJECT_ROOT, "Results", "SA")
    os.makedirs(output_dir, exist_ok=True)

    ResultHandler.save_to_txt(standardized_result, output_dir)
    ResultHandler.save_to_json(standardized_result, output_dir)

    # ===== TRỰC QUAN HÓA BẰNG VISUALIZER CHUNG =====
    print("--- Đang khởi tạo bản đồ trực quan ---")
    try:
        vis_config = config.get('visualization', {})
        vis = Visualizer(
            df_locations,
            osrm_url=vis_config.get('osrm_url', "http://localhost:5001"),
            use_osrm=vis_config.get('use_osrm', True)
        )
        map_path = os.path.join(output_dir, vis_config.get('map_filename', "route_map.html"))
        vis.draw(standardized_result['routes'], map_path)
        print(f"[HOÀN TẤT] Bản đồ lưu tại: {map_path}")
    except Exception as e:
        print(f"[WARNING] Trực quan hóa thất bại: {e}")

    return standardized_result

if __name__ == "__main__":
    run_sa()
