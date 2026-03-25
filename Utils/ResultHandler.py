import os

class ResultHandler:
    @staticmethod
    def save_to_txt(result_data, output_dir):
        """Ghi kết quả từ object chuẩn ra file text"""
        solver_name = result_data['solver_name']
        file_path = os.path.join(output_dir, f"result_{solver_name.lower()}.txt")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"=== BÁO CÁO KẾT QUẢ: {solver_name.upper()} ===\n")
            f.write(f"Quãng đường: {result_data['total_distance_km']:.2f} km\n")
            f.write(f"Thời gian chạy: {result_data['execution_time']:.2f} s\n")
            f.write(f"Số xe sử dụng: {result_data['num_vehicles']}\n")
            f.write("-" * 40 + "\n")
            
            for v_id, route in result_data['routes'].items():
                f.write(f"Xe #{v_id:03d}: {' -> '.join(map(str, route))}\n")
        
        print(f"-> Đã lưu báo cáo tại: {file_path}")