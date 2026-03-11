import pandas as pd
import numpy as np

def load_distance_matrix(file_path):
    """Đọc ma trận khoảng cách từ CSV và chuẩn hóa."""
    try:
        # Load ma trận từ OSRM (đơn vị thường là km hoặc mét)
        data = pd.read_csv(file_path, header=None)
        matrix = data.values
        
        # Scaling: Nhân với 100 và chuyển sang kiểu nguyên (int) 
        # giúp thuật toán chạy nhanh hơn và tránh sai số dấu phẩy động
        matrix_int = np.round(matrix * 100).astype(int)
        return matrix_int
    except Exception as e:
        print(f"Lỗi khi nạp dữ liệu: {e}")
        return None

def generate_initial_solution(num_nodes):
    """
    Tạo lời giải ban đầu: Mỗi xe đi 1 khách hàng rồi quay về kho.
    Đây là lời giải 'tệ' nhất nhưng hợp lệ để thuật toán bắt đầu cải thiện.
    """
    # routes: [[0, 1, 0], [0, 2, 0], ..., [0, 1599, 0]]
    routes = [[0, i, 0] for i in range(1, num_nodes)]
    return routes