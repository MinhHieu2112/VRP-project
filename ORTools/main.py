import json
import os
import time
from Utils.data_loader import DataLoader
from Utils.visualizer import Visualizer
from solver.solver_OR_Tools import ORToolsSolver

def load_config(path='config.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # 1. Load Cấu hình từ JSON
    if not os.path.exists('config.json'):
        print("Lỗi: Không tìm thấy file config.json tại thư mục hiện tại!")
        return
    
    config = load_config()
    
    # Truy xuất các đường dẫn từ JSON mới
    paths = config.get('paths', {})
    output_dir = paths.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Data (DataLoader sẽ tự đọc paths từ config)
    print("--- Đang nạp dữ liệu từ CSV ---")
    loader = DataLoader(config)
    data = loader.load_data()

    # 3. Chạy OR-Tools Solver
    solver_params = config.get('solver_parameters', {})
    time_limit = solver_params.get('time_limit', 300)
    
    print(f"--- Đang bắt đầu giải toán bằng OR-Tools (Limit: {time_limit}s) ---")
    ort_solver = ORToolsSolver(data)
    
    ort_routes, ort_dist = ort_solver.solve(time_limit=time_limit)

    if ort_routes:
        # Lấy scaling_factor từ model_parameters trong JSON
        model_params = config.get('model_parameters', {})
        scaling_factor = model_params.get('scaling_factor', 100)
        
        actual_dist_km = ort_dist / scaling_factor
        
        print(f"Thành công! Tổng quãng đường: {actual_dist_km:.2f} km")

        # 4. XUẤT KẾT QUẢ RA FILE TXT
        txt_path = os.path.join(output_dir, "solver_result.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=== KẾT QUẢ TỐI ƯU HÓA LỘ TRÌNH (OR-TOOLS) ===\n")
            f.write(f"Tổng quãng đường: {actual_dist_km:.2f} km\n")
            f.write(f"Số lượng xe sử dụng: {len(ort_routes)}\n")
            f.write(f"Thời gian giới hạn: {time_limit} giây\n")
            f.write("-" * 30 + "\n")
            f.write("Lộ trình chi tiết:\n")
            
            # Sắp xếp xe theo thứ tự để file txt dễ đọc
            for v_id in sorted(ort_routes.keys()):
                route = ort_routes[v_id]
                # Chỉ lấy các khách hàng (loại bỏ depot số 0)
                nodes_only = [str(node) for node in route if node != 0]
                f.write(f"Route #{v_id}: {' '.join(nodes_only)}\n")
        
        print(f"--- Đã lưu kết quả tại: {txt_path} ---")

        # 5. Vẽ bản đồ trực quan
        print("--- Đang vẽ bản đồ trực quan ---")
        vis_config = config.get('visualization', {})
        viz = Visualizer(data['df_locations'], osrm_url=vis_config.get('osrm_url', "http://localhost:5001"))
        
        map_filename = vis_config.get('map_filename', "route_map.html")
        map_path = os.path.join(output_dir, map_filename)
        
        viz.draw(ort_routes, map_path)
        print(f"--- Bản đồ đã được lưu tại: {map_path} ---")

    else:
        print("Không tìm thấy lời giải trong thời gian quy định hoặc dữ liệu không hợp lệ!")

if __name__ == "__main__":
    main()