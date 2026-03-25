import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, config):
        self.config = config
        # Lấy giá trị từ JSON, nếu không có thì dùng giá trị mặc định (default)
        paths = config.get('paths', {})
        params = config.get('common_model_parameters', {})

        self.locations_path = paths.get('locations_data', '../../Data/locations.csv')
        self.matrix_path = paths.get('distance_matrix', '../../Data/orsm_matrix.csv')
        self.scaling_factor = params.get('scaling_factor', 100)

    def load_matrix(self):
        if not os.path.exists(self.matrix_path):
            raise FileNotFoundError(f"Lỗi: Không tìm thấy file ma trận tại: {self.matrix_path}")
            
        print(f"--- Đang nạp ma trận từ: {self.matrix_path} ---")
        df_matrix = pd.read_csv(self.matrix_path, header=None)
        
        # Chuyển đổi và ép kiểu
        matrix_int = np.round(df_matrix.values * self.scaling_factor).astype(int)
        return matrix_int

    def load_locations(self):
        """Nạp tọa độ Lat, Lng để phục vụ vẽ bản đồ"""
        if not os.path.exists(self.locations_path):
            raise FileNotFoundError(f"Không tìm thấy file khách hàng tại: {self.locations_path}")
            
        df_locs = pd.read_csv(self.locations_path)
        default_demand = self.config.get('global_constraints', {}).get('default_demand', 1)
        # Xử lý nhu cầu (demand) của khách hàng
        if 'demand' not in df_locs.columns:
            # Nếu file CSV không có cột demand, dùng default_demand từ JSON
            # Depot (ID 0) luôn có demand = 0
            demands = np.full(len(df_locs), default_demand, dtype=np.int64)
            demands[0] = 0 
        else:
            demands = df_locs['demand'].values
        return df_locs, demands

    def get_constraints(self):
        constraints = self.config.get('global_constraints', {})
        params = self.config.get('common_model_parameters', {})
        self.default_demand = constraints.get('default_demand', 1)
        
        return {
            "max_vehicles": constraints.get('max_vehicles', 200),
            "vehicle_capacity": constraints.get('vehicle_capacity', 10),
            "num_points": params.get('num_points', 1600),
            "demand": self.default_demand
        }