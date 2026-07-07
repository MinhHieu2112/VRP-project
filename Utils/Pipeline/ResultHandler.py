import os
import json

class ResultHandler:
    @staticmethod
    def save_to_txt(result_data, output_dir):
        """
        Ghi kết quả từ object chuẩn ra file text và xuất thêm bản JSON để Web Demo dễ dàng đọc.
        """
        solver_name = result_data['solver_name']
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Logic tạo tên file tự động tăng (1), (2), (3)... cho lịch sử
        base_filename = f"result_{solver_name.lower().replace(' ', '_')}"
        file_path = os.path.join(output_dir, f"{base_filename}.txt")
        
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(output_dir, f"{base_filename}({counter}).txt")
            counter += 1

        # 2. Xử lý linh hoạt kiểu dữ liệu của routes (list hoặc dict)
        routes_data = result_data['routes']
        if isinstance(routes_data, dict):
            route_values = routes_data.values()
            route_items = routes_data.items()
        else:
            # Nếu là list (như output của thuật toán SA đã sửa)
            route_values = routes_data
            route_items = enumerate(routes_data)

        # 3. Tính tổng số khách hàng được phục vụ
        all_customers = set()
        for route in route_values:
            for node in route:
                if node != 0:
                    all_customers.add(node)

        # 4. Ghi file Text
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

            for v_id, route in route_items:
                num_stops = len([n for n in route if n != 0])
                f.write(f"Xe #{v_id:03d} ({num_stops} điểm): {' -> '.join(map(str, route))}\n")

        print(f"-> Đã lưu báo cáo tại: {file_path}")
        
        # 5. Ghi đè file 'latest_result.json' cho Web UI dễ đọc
        json_path = os.path.join(output_dir, "latest_result.json")
        json_data = {
            "solver_name": result_data['solver_name'],
            "total_distance_km": round(result_data['total_distance_km'], 2),
            "execution_time": round(result_data['execution_time'], 2),
            "num_vehicles": result_data['num_vehicles'],
            "num_customers": len(all_customers),
            "routes": {str(k): v for k, v in result_data['routes'].items()} if isinstance(result_data['routes'], dict) else {str(i): v for i, v in enumerate(result_data['routes'])}
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
            
        print(f"-> Đã lưu JSON cho Web Demo tại: {json_path}")
        return file_path