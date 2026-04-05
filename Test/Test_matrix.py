import numpy as np
import pandas as pd

def analyze_distance_matrix(file_path):
    print("=" * 60)
    print(f"Đang đọc file: {file_path}")
    
    # 1. Load dữ liệu
    df = pd.read_csv(file_path, header=None)
    D = df.values.astype(np.float64)
    
    N = D.shape[0]
    print(f"Kích thước ma trận: {D.shape}")
    
    print("\n" + "=" * 60)
    print("THỐNG KÊ CƠ BẢN")
    print(f"- Min  : {np.min(D)}")
    print(f"- Max  : {np.max(D)}")
    print(f"- Mean : {np.mean(D)}")
    print(f"- Std  : {np.std(D)}")
    
    print("\n" + "=" * 60)
    print("KIỂM TRA LỖI DỮ LIỆU")
    
    # 2. Kiểm tra số âm
    has_negative = np.any(D < 0)
    if has_negative:
        num_negative = np.sum(D < 0)
        min_val = np.min(D)
        print(f"Có số âm!")
        print(f"   - Số lượng: {num_negative}")
        print(f"   - Giá trị nhỏ nhất: {min_val}")
    else:
        print("Không có số âm")
    
    # 3. NaN
    if np.isnan(D).any():
        print(f"Có NaN: {np.isnan(D).sum()} phần tử")
    else:
        print("Không có NaN")
    
    # 4. INF
    if np.isinf(D).any():
        print(f"Có INF: {np.isinf(D).sum()} phần tử")
    else:
        print("Không có INF")
    
    print("\n" + "=" * 60)
    print("KIỂM TRA CẤU TRÚC")
    
    # 5. Đường chéo
    diag_ok = np.allclose(np.diag(D), 0)
    print(f"- Đường chéo = 0: { '✅' if diag_ok else '❌' }")
    
    # 6. Bất đối xứng (ACVRP)
    asymmetry = np.mean(np.abs(D - D.T))
    print(f"- Độ bất đối xứng (mean |D - D^T|): {asymmetry:.6f}")
    
    if asymmetry < 1e-6:
        print("Ma trận gần như đối xứng → có thể là CVRP")
    else:
        print("Ma trận bất đối xứng → đúng ACVRP")
    
    print("\n" + "=" * 60)
    print("PHÂN PHỐI DỮ LIỆU")
    
    percentiles = np.percentile(D, [50, 90, 95, 99])
    print(f"- P50: {percentiles[0]:.4f}")
    print(f"- P90: {percentiles[1]:.4f}")
    print(f"- P95: {percentiles[2]:.4f}")
    print(f"- P99: {percentiles[3]:.4f}")
    
    # 7. Outlier check
    max_val = np.max(D)
    if max_val > percentiles[3] * 10:
        print("Có outlier rất lớn (OSRM có thể lỗi)")
    else:
        print("Không có outlier bất thường")
    
    print("\n" + "=" * 60)
    print("KIỂM TRA KẾT NỐI")
    
    isolated_nodes = []
    for i in range(N):
        if np.all(D[i] > 1e5):
            isolated_nodes.append(i)
    
    if isolated_nodes:
        print(f"Có node bị cô lập: {isolated_nodes[:10]}...")
    else:
        print("Không có node bị cô lập")
    
    print("\n" + "=" * 60)
    print("KẾT LUẬN")
    
    if (not has_negative and 
        not np.isnan(D).any() and 
        not np.isinf(D).any() and 
        diag_ok):
        print("Ma trận hợp lệ để chạy thuật toán VRP")
    else:
        print("Ma trận cần được làm sạch trước khi dùng")
    
    print("=" * 60)


# CHẠY THỬ
if __name__ == "__main__":
    analyze_distance_matrix("../Data/cleaned_matrix_int.csv")