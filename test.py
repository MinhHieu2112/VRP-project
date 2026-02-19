import pandas as pd
import numpy as np

def export_to_ails_format(loc_file, matrix_file, output_name):
    # Đọc dữ liệu
    df_loc = pd.read_csv(loc_file)
    # Nếu file matrix của bạn có header, hãy bỏ header=None hoặc chỉnh lại cho đúng
    matrix = pd.read_csv(matrix_file, header=None).values
    
    n = len(df_loc)
    
    # 1. Tạo Demand giả lập (Demands[0] cho Depot luôn là 0)
    np.random.seed(42) # Giữ kết quả cố định để dễ so sánh nghiên cứu
    demands = np.random.randint(1, 6, size=n)
    demands[0] = 0 
    
    with open(output_name, 'w', encoding='utf-8') as f:
        # --- HEADER ---
        f.write(f"NAME : {output_name}\n")
        f.write("TYPE : CVRP\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("CAPACITY : 20\n") 
        
        # --- 2. NODE_COORD_SECTION (Bắt buộc phải có trước Demand để tránh NullPointer) ---
        f.write("NODE_COORD_SECTION\n")
        for i, row in df_loc.iterrows():
            # ID trong VRP chuẩn thường bắt đầu từ 1
            f.write(f"{i+1} {row['lat']} {row['lon']}\n")
            
        # --- 3. DEMAND_SECTION ---
        f.write("DEMAND_SECTION\n")
        for i, d in enumerate(demands):
            f.write(f"{i+1} {d}\n")
            
        # --- 4. EDGE_WEIGHT_SECTION (Ma trận OSRM) ---
        f.write("EDGE_WEIGHT_SECTION\n")
        for row in matrix:
            # Lưu ý: AILS-II chạy ở chế độ 'rounded true' nên cần số nguyên.
            # Nếu matrix của bạn là mét, int(round(val)) là ổn.
            # Nếu matrix là km, bạn có thể nhân 1000 để đổi sang mét.
            f.write(" ".join(map(lambda x: str(int(round(float(x)))), row)) + "\n")
            
        # --- FOOTER ---
        f.write("DEPOT_SECTION\n1\n-1\n")
        f.write("EOF\n")

    print(f"Đã tạo file {output_name} thành công với {n} tọa độ và ma trận tương ứng.")

# Chạy hàm
export_to_ails_format('Data/locations.csv', 'Data/orsm_matrix.csv', 'my_data_1600.vrp')