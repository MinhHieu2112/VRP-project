import json
import os
import time
from Utils.DataLoaders import DataLoader
from solver.solver_pyVRP import PyVRPSolver

def load_config(file_path='config.json'):
    """Đọc cấu hình từ file JSON."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    print("=== HỆ THỐNG TỐI ƯU HÓA LỘ TRÌNH ACVRP ===")
    
    try:
        # 1. Tải cấu hình
        config = load_config('config.json')
        paths = config.get('paths', {})
        model_params = config.get('model_parameters', {})
        solver_params = config.get('solver_parameters', {})

        # 2. Nạp dữ liệu (Ma trận khoảng cách và Ràng buộc)
        loader = DataLoader(config)
        matrix = loader.load_matrix()
        constraints = loader.get_constraints()

        # 3. Khởi tạo bộ giải PyVRP
        solver = PyVRPSolver(matrix, constraints)

        # 4. Thực hiện tối ưu hóa
        print(f"\n--- Bắt đầu giai đoạn giải thuật ---")
        start_time = time.time()
        
        # Lấy giới hạn thời gian từ JSON, mặc định 180s nếu không có
        time_limit = solver_params.get('time_limit_seconds', 180)
        result = solver.solve(time_limit=time_limit)
        
        end_time = time.time()
        execution_time = end_time - start_time

        # 5. Xử lý và Xuất kết quả
        output_dir = paths.get('output_dir', 'results')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Chuyển đổi khoảng cách từ số nguyên về km (dựa trên scaling_factor)
        scaling_factor = model_params.get('scaling_factor', 100)
        best_distance_km = result.best.distance() / scaling_factor
        num_vehicles = result.best.num_routes()

        print("\n" + "="*40)
        print(f"KẾT QUẢ TỐI ƯU CUỐI CÙNG:")
        print(f" - Tổng quãng đường: {best_distance_km:.2f} km")
        print(f" - Số xe sử dụng: {num_vehicles}")
        print(f" - Thời gian thực thi: {execution_time:.2f} giây")
        print("="*40)

        # Ghi kết quả ra file
        result_filename = paths.get('result_file', 'ket_qua_lo_trinh.txt')
        output_path = os.path.join(output_dir, result_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"--- BÁO CÁO KẾT QUẢ ACVRP ---\n")
            f.write(f"Thời điểm chạy: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Cấu hình: {json.dumps(config, indent=2)}\n")
            f.write(f"--------------------------------\n")
            f.write(f"Tổng quãng đường: {best_distance_km:.2f} km\n")
            f.write(f"Số xe thực tế sử dụng: {num_vehicles}\n")
            f.write(f"Thời gian tính toán: {execution_time:.2f} giây\n\n")
            f.write("Chi tiết các tuyến đường:\n")
            f.write(str(result.best))

        print(f"\n==> Thành công! Kết quả chi tiết đã được lưu tại: {output_path}")

    except Exception as e:
        print(f"\n[LỖI HỆ THỐNG]: {str(e)}")

if __name__ == "__main__":
    main()