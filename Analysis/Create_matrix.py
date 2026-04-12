import pandas as pd
import numpy as np

def split_orsm_matrix(input_file, scales):
    """
    Yêu cầu 1: Tách ma trận OSRM gốc thành các quy mô nhỏ hơn.
    """
    try:
        # Đọc ma trận gốc (giả định file không có header hoặc header là index)
        df = pd.read_csv(input_file, header=None)
        
        for n in scales:
            if n <= len(df):
                # Tách lấy n hàng và n cột đầu tiên
                sub_matrix = df.iloc[:n, :n]
                output_name = f"orsm_matrix_{n}.csv"
                sub_matrix.to_csv(output_name, index=False, header=False)
                print(f"Đã tạo: {output_name} với kích thước {n}x{n}")
            else:
                print(f"Cảnh báo: Quy mô {n} lớn hơn kích thước ma trận gốc ({len(df)})")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {input_file}")

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Hàm tính khoảng cách Haversine giữa các điểm (đơn vị: mét).
    """
    R = 6371000  # Bán kính Trái đất tính bằng mét
    
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def create_haversine_matrix(locations_file, target_size=1600):
    """
    Yêu cầu 2: Giả lập ma trận 1600 điểm dựa trên tọa độ Lat/Lon.
    """
    try:
        # Đọc file tọa độ (giả định có cột 'lat' và 'lng' hoặc 'latitude' và 'longitude')
        df_locs = pd.read_csv(locations_file)
        
        # Nếu số lượng điểm trong file ít hơn 1600, ta sẽ lặp lại/giả lập thêm
        if len(df_locs) < target_size:
            print(f"Dữ liệu gốc có {len(df_locs)} điểm. Đang nhân bản để đủ {target_size}...")
            repeats = (target_size // len(df_locs)) + 1
            df_locs = pd.concat([df_locs] * repeats, ignore_index=True)
        
        df_target = df_locs.iloc[:target_size].reset_index(drop=True)
        
        # Lấy mảng numpy để tính toán vector hóa
        lats = df_target['lat'].values
        lons = df_target['lon'].values
        
        # Tạo ma trận khoảng cách
        # Sử dụng broadcasting của numpy để tính toán nhanh 1600x1600
        dist_matrix = haversine_vectorized(
            lats[:, np.newaxis], lons[:, np.newaxis],
            lats[np.newaxis, :], lons[np.newaxis, :]
        )
        
        # Chuyển thành đơn vị nội bộ (nếu bạn muốn giống OSRM: chia 10 và làm tròn)
        # Ở đây lưu đơn vị mét chuẩn.
        pd.DataFrame(dist_matrix).to_csv("haversine_matrix.csv", index=False, header=False)
        print(f"Đã tạo: haversine_matrix.csv với kích thước {target_size}x{target_size}")

    except Exception as e:
        print(f"Lỗi khi tạo ma trận Haversine: {e}")

if __name__ == "__main__":
    # 1. Tách ma trận OSRM
    # Lưu ý: Thay 'orsm_matrix.csv' bằng đường dẫn thực tế của bạn
    split_orsm_matrix("../Data/orsm_matrix.csv", [200, 500, 1000])
    
    # 2. Tạo ma trận Haversine 1600x1600
    # Lưu ý: Đảm bảo locations.csv có cột 'lat' và 'lng'
    create_haversine_matrix("../Data/locations.csv", 1600)