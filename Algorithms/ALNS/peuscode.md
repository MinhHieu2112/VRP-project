
## Adaptive Large Neighborhood Search (ALNS)

```pascal
Initialize S using initial strategy
Initialize unassigned ← empty list
Initialize destroy_operators ← {random_removal, worst_removal}
Initialize repair_operators ← {greedy_insertion, regret_insertion}
Initialize weights_destroy, weights_repair ← equal weights
Initialize scores_destroy, scores_repair ← 0

Initialize S* ← S
Initialize best_cost ← objective(S)
Initialize T ← T_start (for Simulated Annealing acceptance)

for iteration = 1 to max_iterations do
    # --- Chọn toán tử phá hủy và tái thiết bằng Roulette Wheel ---
    destroy_op ← RouletteWheelSelect(destroy_operators, weights_destroy)
    repair_op  ← RouletteWheelSelect(repair_operators, weights_repair)

    # --- Tạo nghiệm lân cận bằng Destroy & Repair ---
    S_removed, removed_nodes ← Apply destroy_op(S)
    S_new ← Apply repair_op(S_removed, removed_nodes)

    Δ = objective(S_new) - objective(S)

    # --- Đánh giá chấp nhận nghiệm theo Simulated Annealing ---
    if Δ < 0 or random() < exp(-Δ / T) then
        S ← S_new
        
        # Xác định điểm thưởng (Score) cho toán tử
        if objective(S_new) < best_cost then
            S* ← S_new
            best_cost ← objective(S_new)
            score ← score_best_found (e.g., 15)
        else if Δ < 0 then
            score ← score_improving (e.g., 8)
        else
            score ← score_accepted (e.g., 6)
        end if
    else
        score ← score_rejected (e.g., 0)
    end if

    # --- Cập nhật điểm và trọng số toán tử ---
    UpdateScores(destroy_op, repair_op, score)
    if iteration % update_period == 0 then
        UpdateWeights(weights_destroy, weights_repair, scores_destroy, scores_repair, decay)
        ResetScores(scores_destroy, scores_repair)
    end if

    T = T * alpha  # Hạ nhiệt độ
end for

# --- Hậu tối ưu ---
S* ← apply_2opt_or_opt_to_all_routes(S*)
return S*, objective(S*)
```
