import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, config):
        self.config = config
        # Lấy giá trị từ JSON, nếu không có thì dùng giá trị mặc định (default)
        paths = config.get('paths', {})
        params = config.get('model_parameters', {})
        
        self.matrix_path = paths.get('distance_matrix', '../Data/orsm_matrix.csv')
        self.scaling_factor = params.get('scaling_factor', 100)

    def load_matrix(self):
        if not os.path.exists(self.matrix_path):
            raise FileNotFoundError(f"Lỗi: Không tìm thấy file ma trận tại: {self.matrix_path}")
            
        print(f"--- Đang nạp ma trận từ: {self.matrix_path} ---")
        df_matrix = pd.read_csv(self.matrix_path, header=None)
        
        # Chuyển đổi và ép kiểu
        matrix_int = np.round(df_matrix.values * self.scaling_factor).astype(int)
        return matrix_int

    def get_constraints(self):
        constraints = self.config.get('constraints', {})
        params = self.config.get('model_parameters', {})
        
        return {
            "max_vehicles": constraints.get('max_vehicles', 200),
            "vehicle_capacity": constraints.get('vehicle_capacity', 10),
            "num_points": params.get('num_points', 1600)
        }