import os
import json
from Utils.Data_loader import DataLoader
from Utils.Visualizer import Visualizer
from Algorithms.ORTools.solver.solver_OR_Tools import ORToolsSolver

CONFIG_PATH = 'Algorithms/ORTools/config.json'

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy cấu hình tại: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_result_to_txt(output_path, routes, actual_dist_km, time_limit):
    """Ghi kết quả tối ưu hóa ra file văn bản."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== KẾT QUẢ TỐI ƯU HÓA LỘ TRÌNH (OR-TOOLS) ===\n")
        f.write(f"Tổng quãng đường: {actual_dist_km:.2f} km\n")
        f.write(f"Số lượng xe sử dụng: {len(routes)}\n")
        f.write(f"Thời gian giới hạn: {time_limit} giây\n")
        f.write("-" * 40 + "\n")
        f.write("Lộ trình chi tiết (Danh sách khách hàng):\n")
        
        for v_id in sorted(routes.keys()):
            # Loại bỏ depot (node 0) để chỉ hiển thị mã khách hàng
            route_str = " -> ".join(str(node) for node in routes[v_id])
            f.write(f"Route #{v_id:03d}: {route_str}\n")

def run_or_tools():
    try:
        # 1. Khởi tạo cấu hình
        config = load_config(CONFIG_PATH)
        paths = config['paths']
        solver_cfg = config['solver_parameters']
        vis_cfg = config['visualization']
        
        os.makedirs(paths['output_dir'], exist_ok=True)

        # 2. Nạp dữ liệu
        print(f"--- [1/4] Nạp dữ liệu từ: {paths['distance_matrix']} ---")
        loader = DataLoader(config)
        data = loader.load_data()

        # 3. Thực thi bộ giải (Pass toàn bộ config vào solver)
        print(f"--- [2/4] Đang giải toán (Limit: {solver_cfg['time_limit']}s) ---")
        solver = ORToolsSolver(data, config)
        routes, actual_dist_km = solver.solve() # Logic scaling đã nằm trong solver

        if not routes:
            print("(!) Không tìm thấy lời giải hợp lệ.")
            return

        # 4. Xuất kết quả & Trực quan hóa
        print(f"--- [3/4] Đang lưu kết quả. Tổng quãng đường: {actual_dist_km:.2f} km ---")
        
        # Lưu file TXT
        txt_out = os.path.join(paths['output_dir'], "solver_result.txt")
        save_result_to_txt(txt_out, routes, actual_dist_km, solver_cfg['time_limit'])
        
        # Vẽ bản đồ
        print(f"--- [4/4] Đang vẽ bản đồ: {vis_cfg['map_filename']} ---")
        viz = Visualizer(data['df_locations'], osrm_url=vis_cfg['osrm_url'])
        map_out = os.path.join(paths['output_dir'], vis_cfg['map_filename'])
        viz.draw(routes, map_out)

        print("\n[HOÀN TẤT] Mọi dữ liệu đã được lưu tại thư mục Results.")

    except Exception as e:
        print(f"[LỖI HỆ THỐNG]: {e}")

if __name__ == "__main__":
    run_or_tools()