import pandas as pd
import re
import glob
import os

def process_all_results(results_dir_pattern, matrix_file_path):
    # 1. Nạp ma trận duy nhất một lần (Optimization)
    print(f"--- Đang nạp ma trận: {os.path.basename(matrix_file_path)} ---")
    df_matrix = pd.read_csv(matrix_file_path, header=None)
    matrix = df_matrix.values
    
    summary_results = []
    
    # 2. Quét tất cả file thỏa mãn pattern (ví dụ: ../Results/**/*.txt)
    file_list = glob.glob(results_dir_pattern, recursive=True)
    
    if not file_list:
        print("Không tìm thấy file kết quả nào!")
        return

    print(f"Tìm thấy {len(file_list)} file. Bắt đầu tính toán...\n")
    print(f"{'Tên File':<40} | {'Quãng đường (km)':>15}")
    print("-" * 60)

    for file_path in sorted(file_list):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Regex đã fix lỗi 'bad character range' của bạn
        route_lines = re.findall(r'Xe #\d+.*?: ([\d\s\->]+)', content)
        
        total_dist_meters = 0
        for line in route_lines:
            nodes = [int(node.strip()) for node in line.split('->')]
            for i in range(len(nodes) - 1):
                u, v = nodes[i], nodes[i+1]
                total_dist_meters += matrix[u][v]
        
        total_km = total_dist_meters / 1000
        file_name = os.path.basename(file_path)
        
        summary_results.append((file_name, total_km))
        print(f"{file_name:<40} | {total_km:>15.2f}")

    return summary_results

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Dấu ** giúp tìm trong tất cả thư mục con (recursive)
PATTERN = '../Results/ACO/ClarkeWright-1600/*.txt'  
MATRIX_PATH = '../Data/orsm_matrix.csv'

# Chạy lệnh
results = process_all_results(PATTERN, MATRIX_PATH)