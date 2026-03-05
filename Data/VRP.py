import pandas as pd
import numpy as np

def export_for_lkh3_fixed_v2(loc_file, matrix_file, output_name):
    print(f"--- Đang tạo file chuẩn LKH-3: {output_name} ---")
    
    # 1. Đọc dữ liệu
    df_loc = pd.read_csv(loc_file)
    if matrix_file.endswith('.npy'):
        matrix = np.load(matrix_file)
    else:
        matrix = pd.read_csv(matrix_file, header=None).values

    n = matrix.shape[0]

    with open(output_name, 'w', encoding='utf-8') as f:
        # Header - Bỏ đuôi .vrp ở phần NAME cho sạch
        f.write(f"NAME : my_data_1600\n")
        f.write("TYPE : ACVRP\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("CAPACITY : 10\n") 
        
        # --- QUAN TRỌNG: EDGE_WEIGHT_SECTION PHẢI Ở ĐÂY ---
        f.write("EDGE_WEIGHT_SECTION\n")
        for row in matrix:
            for val in row:
                # Ghi mỗi số 1 dòng để an toàn tuyệt đối
                val_int = int(round(float(val) * 100))
                f.write(f"{val_int}\n")

        # --- SAU ĐÓ MỚI ĐẾN DEMAND_SECTION ---
        f.write("DEMAND_SECTION\n")
        for i in range(n):
            demand = 0 if i == 0 else 1
            f.write(f"{i+1} {demand}\n")
            
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")

    print(f"✅ Đã sửa lại thứ tự Section thành công!")

export_for_lkh3_fixed_v2('Data/locations.csv', 'Data/orsm_matrix.csv', 'my_data_1600.vrp')