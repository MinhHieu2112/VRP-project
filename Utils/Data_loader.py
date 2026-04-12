"""
Utils/Data_loader.py
FIXES:
  [FIX-4] realpath(__file__) thay abspath: khi import qua sys.path.append từ thư mục con,
           __file__ có thể là relative → abspath resolve từ cwd (sai). realpath luôn đúng.
  [FIX-8] Đọc cả 'locations_data' (Utils/config) và 'locations' (ACO/config).
"""

import pandas as pd
import numpy as np
import os

_THIS_FILE   = os.path.realpath(__file__)          # [FIX-4] realpath, not abspath
_UTILS_DIR   = os.path.dirname(_THIS_FILE)
PROJECT_ROOT = os.path.dirname(_UTILS_DIR)


class DataLoader:
    KM_SCALE = 100  # 1 unit = 10m, 1 km = 100 units

    def __init__(self, config: dict):
        self.config = config
        paths = config.get('paths', {})
        matrix_rel   = paths.get('distance_matrix', 'Data/orsm_matrix.csv')
        # [FIX-8] Tương thích cả hai key name
        location_rel = (paths.get('locations_data') or
                        paths.get('locations', 'Data/locations.csv'))
        self.matrix_path   = os.path.normpath(os.path.join(PROJECT_ROOT, matrix_rel))
        self.customer_path = os.path.normpath(os.path.join(PROJECT_ROOT, location_rel))

    def load_data(self) -> dict:
        constraints      = self.config.get('global_constraints', {})
        num_vehicles     = constraints.get('max_vehicles',    200)
        vehicle_capacity = constraints.get('vehicle_capacity',  10)
        default_demand   = constraints.get('default_demand',     1)
        matrix_int       = self._load_matrix()
        df_locs, demands = self._load_locations(default_demand)
        return {
            "distance_matrix":   matrix_int,
            "num_vehicles":      num_vehicles,
            "depot":             0,
            "demands":           demands,
            "vehicle_capacity":  vehicle_capacity,
            "df_locations":      df_locs,
        }

    def _load_matrix(self) -> np.ndarray:
        if not os.path.exists(self.matrix_path):
            raise FileNotFoundError(
                f"Không tìm thấy ma trận: {self.matrix_path}\n"
                f"  PROJECT_ROOT = {PROJECT_ROOT}")
        df  = pd.read_csv(self.matrix_path, header=None)
        raw = np.nan_to_num(df.values.astype(float), nan=0.0)
        mat = np.round(raw / 10.0).astype(np.int64)
        mat = np.clip(mat, 0, None)
        print(f"[DataLoader] Ma trận {mat.shape} | "
              f"min={mat.min()} max={mat.max()} (đơn vị nội bộ, 1 unit = 10m)")
        return mat

    def _load_locations(self, default_demand: int):
        if not os.path.exists(self.customer_path):
            raise FileNotFoundError(
                f"Không tìm thấy tọa độ: {self.customer_path}\n"
                f"  PROJECT_ROOT = {PROJECT_ROOT}")
        df = pd.read_csv(self.customer_path)
        if 'demand' in df.columns:
            demands = df['demand'].values.astype(np.int64)
        else:
            demands    = np.full(len(df), default_demand, dtype=np.int64)
            demands[0] = 0
        return df, demands