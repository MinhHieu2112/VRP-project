import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, config):
        """
        Khởi tạo DataLoader bằng đối tượng config từ JSON.
        """
        self.config = config
        # Lấy đường dẫn từ config, nếu không có thì dùng giá trị mặc định
        self.matrix_path = config.get('paths', {}).get('distance_matrix', 'Data/orsm_matrix.csv')
        self.customer_path = config.get('paths', {}).get('locations', 'Data/locations.csv')

    def load_data(self):
        """
        Đọc dữ liệu dựa trên các tham số cấu hình trong JSON.
        """
        # 1. Lấy các tham số ràng buộc từ JSON
        constraints = self.config.get('constraints', {})
        num_vehicles = constraints.get('max_vehicles', 200)
        vehicle_capacity = constraints.get('vehicle_capacity', 10)
        default_demand = constraints.get('default_demand', 1)
        scaling_factor = self.config.get('scaling_factor', 100)

        # 2. Đọc ma trận khoảng cách
        if not os.path.exists(self.matrix_path):
            raise FileNotFoundError(f"Không tìm thấy file ma trận tại: {self.matrix_path}")
            
        df_matrix = pd.read_csv(self.matrix_path, header=None)
        matrix_float = np.nan_to_num(df_matrix.values.astype(float), nan=0.0)
        
        # Thực hiện Scaling (ví dụ từ 1.23 km thành 123 đơn vị nguyên)
        matrix_int = np.round(matrix_float * scaling_factor).astype(np.int64)
        matrix_int[matrix_int < 0] = 0

        # 3. Đọc file tọa độ khách hàng
        if not os.path.exists(self.customer_path):
            raise FileNotFoundError(f"Không tìm thấy file khách hàng tại: {self.customer_path}")
            
        df_locs = pd.read_csv(self.customer_path)
        
        # Xử lý nhu cầu (demand) của khách hàng
        if 'demand' not in df_locs.columns:
            # Nếu file CSV không có cột demand, dùng default_demand từ JSON
            # Depot (ID 0) luôn có demand = 0
            demands = np.ones(len(df_locs)) * default_demand
            demands[0] = 0 
        else:
            demands = df_locs['demand'].values

        return {
            "distance_matrix": matrix_int,
            "scaling_factor": scaling_factor,
            "num_vehicles": num_vehicles,
            "depot": 0,
            "demands": demands,
            "vehicle_capacity": vehicle_capacity,
            "df_locations": df_locs
        }