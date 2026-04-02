import os
import json

class ResultHandler:
    @staticmethod
    def save_to_txt(result_data, output_dir):
        """
        Ghi kết quả từ object chuẩn ra file text.

        Format chuẩn của result_data:
        {
            "solver_name": str,
            "total_distance_km": float,
            "execution_time": float (seconds),
            "routes": dict[int, list[int]],  # {0: [0,3,7,0], ...}
            "num_vehicles": int
        }
        """
        solver_name = result_data['solver_name']
        file_path = os.path.join(output_dir, f"result_{solver_name.lower().replace(' ', '_')}.txt")

        os.makedirs(output_dir, exist_ok=True)

        # Tính tổng số khách hàng được phục vụ
        all_customers = set()
        for route in result_data['routes'].values():
            for node in route:
                if node != 0:
                    all_customers.add(node)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"{'='*50}\n")
            f.write(f"  BÁO CÁO KẾT QUẢ: {solver_name.upper()}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Tổng quãng đường:     {result_data['total_distance_km']:.2f} km\n")
            f.write(f"Thời gian chạy:       {result_data['execution_time']:.2f} s\n")
            f.write(f"Số xe sử dụng:        {result_data['num_vehicles']}\n")
            f.write(f"Số khách hàng:        {len(all_customers)}\n")
            f.write(f"{'─'*50}\n")
            f.write(f"CHI TIẾT LỘ TRÌNH:\n")
            f.write(f"{'─'*50}\n")

            for v_id, route in result_data['routes'].items():
                num_stops = len([n for n in route if n != 0])
                f.write(f"Xe #{v_id:03d} ({num_stops} điểm): {' -> '.join(map(str, route))}\n")

        print(f"-> Đã lưu báo cáo tại: {file_path}")

    @staticmethod
    def save_to_json(result_data, output_dir):
        """
        Lưu kết quả ra file JSON để dễ so sánh giữa các thuật toán.
        """
        solver_name = result_data['solver_name']
        file_path = os.path.join(output_dir, f"result_{solver_name.lower().replace(' ', '_')}.json")

        os.makedirs(output_dir, exist_ok=True)

        # Convert dict keys sang string cho JSON
        json_data = {
            "solver_name": result_data['solver_name'],
            "total_distance_km": round(result_data['total_distance_km'], 2),
            "execution_time": round(result_data['execution_time'], 2),
            "num_vehicles": result_data['num_vehicles'],
            "routes": {str(k): v for k, v in result_data['routes'].items()}
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"-> Đã lưu JSON tại: {file_path}")