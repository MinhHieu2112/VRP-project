def calculate_gaps(best_val_of_best, current_val_of_best, best_val_of_avg, current_val_of_avg):
    """
    Tính toán độ lệch phần trăm (Gap).
    Công thức: Gap (%) = ((Giá trị xét - Giá trị tốt nhất) / Giá trị tốt nhất) * 100
    """
    
    # Tính %Best Gap
    # So sánh nghiệm tốt nhất của thuật toán hiện tại với nghiệm tốt nhất toàn cục
    best_gap = ((current_val_of_best - best_val_of_best) / best_val_of_best) * 100
    
    # Tính %Avg Gap
    # So sánh nghiệm trung bình của thuật toán hiện tại với nghiệm trung bình tốt nhất (hoặc baseline)
    avg_gap = ((current_val_of_avg - best_val_of_avg) / best_val_of_avg) * 100
    
    return {
        "Best Gap (%)": round(best_gap, 4),
        "Avg Gap (%)": round(avg_gap, 4)
    }

# Ví dụ chạy thử:
# Giả sử PyVRP cho Best là 250.0, thuật toán của bạn cho Best là 254.26
# Giả sử PyVRP cho Avg là 252.0, thuật toán của bạn cho Avg là 258.0
results = calculate_gaps(102.88, 115.98, 103.02, 118.87)
print(results)