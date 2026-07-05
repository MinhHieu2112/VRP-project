import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MatrixCleaner:
    def __init__(self, matrix_path: str, loc_path: str, precision_factor: int = 1):
        self.matrix_path = Path(matrix_path)
        self.loc_path = Path(loc_path)
        self.precision_factor = precision_factor
        self.eps = 1e-6

    def load_data(self):
        logger.info("Đang tải dữ liệu...")
        # Dùng numpy để load ma trận nhanh hơn pandas
        self.D = np.genfromtxt(self.matrix_path, delimiter=',')
        df_loc = pd.read_csv(self.loc_path)
        self.coords = df_loc[['lon', 'lat']].values
        self.N = len(self.coords)
        
        if self.D.shape[0] != self.N:
            raise ValueError(f"Kích thước không khớp: Matrix {self.D.shape}, Locs {self.N}")

    def _vectorized_haversine(self, i_indices, j_indices):
        """Tính Haversine cho danh sách các cặp index (vectorized)"""
        R = 6371000  # Bán kính Trái Đất (mét)
        
        p1 = np.radians(self.coords[i_indices])
        p2 = np.radians(self.coords[j_indices])
        
        dlat = p2[:, 1] - p1[:, 1]
        dlon = p2[:, 0] - p1[:, 0]
        
        a = np.sin(dlat/2)**2 + np.cos(p1[:, 1]) * np.cos(p2[:, 1]) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def clean(self, penalty_factor: float = 2.0):
        logger.info("🧹 Bắt đầu làm sạch ma trận...")
        
        # 1. Xử lý giá trị âm/NaN/Inf bằng Haversine + Penalty
        invalid_mask = (self.D < 0) | np.isnan(self.D) | np.isinf(self.D)
        num_invalid = np.sum(invalid_mask)
        
        if num_invalid > 0:
            logger.warning(f"Phát hiện {num_invalid} điểm lỗi. Đang sửa bằng Haversine x{penalty_factor}...")
            i_idx, j_idx = np.where(invalid_mask)
            replacement_values = self._vectorized_haversine(i_idx, j_idx) * penalty_factor
            self.D[invalid_mask] = replacement_values

        # 2. Xử lý đường chéo và noise
        np.fill_diagonal(self.D, 0)
        self.D[self.D < self.eps] = 0
        
        return self

    def to_integer(self):
        logger.info(f"Chuyển đổi sang số nguyên (Factor: {self.precision_factor})...")
        # Nhân với hệ số tỷ lệ để giữ độ chính xác sau dấu phẩy nếu cần
        D_scaled = self.D * self.precision_factor
        return np.rint(D_scaled).astype(np.int64)

    def save(self, output_path: str, data: np.ndarray):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Lưu file tối ưu: không index, không header, định dạng số nguyên
        np.savetxt(output_path, data, delimiter=',', fmt='%d')
        logger.info(f"Đã lưu ma trận sạch tại: {output_path}")

# ==============================
# EXECUTION
# ==============================
if __name__ == "__main__":
    # Khởi tạo cleaner
    cleaner = MatrixCleaner(
        matrix_path="../Data/haversine_matrix.csv",
        loc_path="../Data/locations.csv",
        precision_factor=1 
    )

    try:
        cleaner.load_data()
        # Dùng penalty 2.0 để các đoạn đường lỗi "đắt" gấp đôi đường chim bay
        cleaner.clean(penalty_factor=2.0)
        final_matrix = cleaner.to_integer()
        
        cleaner.save("../Data/cleaned_matrix_int.csv", final_matrix)
        
        # Kiểm tra nhanh
        logger.info(f"Thống kê: Min={final_matrix.min()}, Max={final_matrix.max()}")
    except Exception as e:
        logger.error(f"Quá trình thất bại: {e}")