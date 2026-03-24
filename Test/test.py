import re
import os
import pandas as pd # Đảm bảo bạn đã cài đặt pandas: pip install pandas

def extract_routes_from_text(file_path):
    """
    Bóc tách các tuyến đường từ file txt, xử lý các tag hoặc.
    """
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file kết quả tại {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Loại bỏ các tag nhiễu thường gặp trong dữ liệu
        clean_content = re.sub(r'\\', '', content)
        clean_content = re.sub(r'\\', '', clean_content)
        
        # Tìm các đoạn nội dung sau "Route #x:"
        route_patterns = re.findall(r'Route\s*#\d+:(.*?)(?=Route\s*#|$)', clean_content, re.DOTALL)
        
        routes = []
        for p in route_patterns:
            nodes = [int(node) for node in p.split() if node.strip().isdigit()]
            if nodes:
                routes.append(nodes)
        return routes
    except Exception as e:
        print(f"Lỗi khi xử lý file văn bản: {e}")
        return []

def load_distance_matrix(csv_path):
    """
    Đọc ma trận khoảng cách từ file CSV.
    Giả định file CSV không có header và các hàng/cột tương ứng với ID của node.
    """
    if not os.path.exists(csv_path):
        print(f"Lỗi: Không tìm thấy file ma trận tại {csv_path}")
        return None

    try:
        print(f"--- Đang nạp ma trận từ {csv_path} ---")
        # Đọc CSV không có tiêu đề, chuyển thành mảng 2 chiều
        df = pd.read_csv(csv_path, header=None)
        return df.values
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return None

def verify_optimization(routes, distance_matrix, depot_id=0):
    """
    Tính toán lại tổng quãng đường bằng cách cộng dồn giá trị từ ma trận.
    """
    total_distance = 0.0
    
    for i, route in enumerate(routes):
        route_dist = 0.0
        # Xe đi từ Kho -> Lộ trình khách hàng -> Quay về Kho
        full_path = [depot_id] + route + [depot_id]
        
        for j in range(len(full_path) - 1):
            u, v = full_path[j], full_path[j+1]
            try:
                dist = distance_matrix[u][v]
                route_dist += dist
            except IndexError:
                print(f"Lỗi: Node {u} hoặc {v} vượt quá kích thước ma trận ({len(distance_matrix)}x{len(distance_matrix)}).")
                return None
        
        total_distance += route_dist
        if i < 3: # In thử 3 lộ trình đầu tiên để kiểm soát
            print(f"Xe #{i+1:03d} | Số điểm: {len(route):02d} | Quãng đường: {route_dist:.2f} km")
            
    return total_distance

if __name__ == "__main__":
    # 1. Cấu hình đường dẫn (Điều chỉnh nếu cần thiết)
    RESULT_TXT = '../Results/ORTools/solver_result.txt' 
    MATRIX_CSV = '../Data/orsm_matrix.csv'
    DEPOT = 0 

    # 2. Thực hiện trích xuất lộ trình
    print("--- BẮT ĐẦU QUY TRÌNH XÁC THỰC ---")
    routes = extract_routes_from_text(RESULT_TXT)
    
    if not routes:
        print("Không tìm thấy lộ trình hợp lệ để xử lý.")
    else:
        # 3. Nạp ma trận thực tế
        actual_matrix = load_distance_matrix(MATRIX_CSV)
        
        if actual_matrix is not None:
            print(f"Đã nạp ma trận. Kích thước: {len(actual_matrix)}x{len(actual_matrix[0])}")
            print(f"Đã trích xuất: {len(routes)} xe từ file kết quả.")
            
            # 4. Tính toán và đối chiếu
            print("\n--- CHI TIẾT TÍNH TOÁN LẠI ---")
            recalculated_dist = verify_optimization(routes, actual_matrix, DEPOT)
            
            if recalculated_dist is not None:
                print("-" * 45)
                print(f"KẾT QUẢ CUỐI CÙNG:")
                print(f" - Tổng số xe sử dụng: {len(routes)}")
                print(f" - Tổng quãng đường tính lại: {recalculated_dist:.2f} km")
                print("-" * 45)