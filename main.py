import os
import json
import time
import traceback

# Chỉ import Utils chung ở đây
from Utils.Data_loader import DataLoader
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer

# Import các Solvers
from Algorithms.ORTools.solver.solver_OR_Tools import ORToolsSolver
from Algorithms.PyVRP.solver.solver_pyVRP import PyVRPSolver

CONFIG_PATH = 'Utils/config.json'

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy cấu hình tại: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_experiment(solver_type):
    """
    Quy trình thống nhất cho mọi bộ giải:
    Load Data -> Solve -> Standardize -> Save & Visualize
    """
    print(f"\n{'='*20} KHỞI CHẠY THỰC NGHIỆM: {solver_type.upper()} {'='*20}")
    
    try:
        # 1. Khởi tạo cấu hình và dữ liệu (DÙNG CHUNG)
        config = load_config(CONFIG_PATH)
        loader = DataLoader(config)
        data_bundle = loader.load_data() 
        
        # Lấy thông số solver từ config
        solver_cfg = config['solvers'].get(solver_type, {})
        time_limit = solver_cfg.get('time_limit_seconds', 300)
        
        # 2. Khởi tạo và chạy bộ giải
        start_time = time.time()
        standardized_result = {}

        if solver_type == "py_vrp":
            # PyVRP cần ma trận và ràng buộc
            solver = PyVRPSolver(data_bundle['distance_matrix'], config['global_constraints'])
            res = solver.solve(time_limit=time_limit)
            
            # ADAPTER: Chuyển PyVRP format sang Standard format
            # Lấy list các visits từ mỗi route và thêm Depot (0) vào đầu/cuối
            routes_dict = {}
            for i, route in enumerate(res.best.routes()):
                routes_dict[i] = [0] + route.visits() + [0]
            
            total_distance = res.best.distance() 
            scaling_factor = config['common_model_parameters']['scaling_factor']
            standardized_result = {
                "solver_name": "PyVRP",
                "total_distance_km": total_distance / scaling_factor,
                "execution_time": time.time() - start_time,
                "routes": routes_dict,
                "num_vehicles": len(routes_dict)
            }

        elif solver_type == "or_tools":
            solver = ORToolsSolver(data_bundle, config)
            routes, dist_km = solver.solve() # Solver này bạn đã viết đo time bên trong hoặc đo ở ngoài
            
            standardized_result = {
                "solver_name": "OR-Tools",
                "total_distance_km": dist_km,
                "execution_time": time.time() - start_time,
                "routes": routes,
                "num_vehicles": len(routes)
            }

        # 3. Hậu xử lý tập trung (DÙNG CHUNG)
        if not standardized_result.get("routes"):
            print(f"(!) {solver_type} không tìm thấy lời giải.")
            return

        # Tạo thư mục con cho từng solver trong Results
        output_base = config['paths']['output_dir']
        solver_output_dir = os.path.join(output_base, solver_type)
        os.makedirs(solver_output_dir, exist_ok=True)

        # A. Ghi báo cáo TXT
        ResultHandler.save_to_txt(standardized_result, solver_output_dir)

        # B. Vẽ bản đồ (Visualizer dùng chung df_locations nạp từ loader)
        print(f"--- Đang trực quan hóa lộ trình ---")
        try:
            viz = Visualizer(
                data_bundle['df_locations'], 
                osrm_url=config['visualization'].get('osrm_url', "http://localhost:5001"),
                use_osrm=config['visualization'].get('use_osrm', True)
            )
            map_path = os.path.join(solver_output_dir, config['visualization']['map_filename'])
            viz.draw(standardized_result['routes'], map_path)
            print(f"[HOÀN TẤT] Map lưu tại: {map_path}")
        except Exception as e:
            print(f"[WARNING] Trực quan hóa thất bại: {e}")

        print(f"\n[HOÀN TẤT] Kết quả {solver_type} lưu tại: {solver_output_dir}")

    except Exception as e:
        print(f"\n[LỖI HỆ THỐNG]: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # Bạn có thể chạy lần lượt hoặc chạy cả 2 để so sánh
    # run_experiment("or_tools")
    run_experiment("py_vrp")