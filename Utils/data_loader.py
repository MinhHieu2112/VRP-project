import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, matrix_path, customer_path):
        self.matrix_path = matrix_path
        self.customer_path = customer_path

    def load_data(self, num_vehicles=200, vehicle_capacity=10):
        # 1. Đọc ma trận khoảng cách và thực hiện Scaling 100
        df_matrix = pd.read_csv(self.matrix_path, header=None)
        matrix_float = np.nan_to_num(df_matrix.values.astype(float), nan=0.0)
        
        scaling_factor = 100
        matrix_int = np.round(matrix_float * scaling_factor).astype(np.int64)
        matrix_int[matrix_int < 0] = 0

        # 2. Đọc file khách hàng
        df_locs = pd.read_csv(self.customer_path)
        
        # Tạo demand mặc định nếu chưa có
        if 'demand' not in df_locs.columns:
            demands = [0] + [1] * (len(df_locs) - 1)
        else:
            demands = df_locs['demand'].tolist()

        return {
            "distance_matrix": matrix_int.tolist(),
            "scaling_factor": scaling_factor,
            "num_vehicles": num_vehicles,
            "depot": 0,
            "demands": demands,
            "vehicle_capacities": [vehicle_capacity] * num_vehicles,
            "df_locations": df_locs  # Dùng cho visualizer
        }