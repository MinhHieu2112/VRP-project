
## Granular Tabu Search (GTS) - Tích hợp PenaltyController và Candidate List

```pascal
Initialize S using initial strategy (e.g., random/greedy)
Initialize C ← array of individual route costs for each route in S
Initialize L ← array of individual route loads for each route in S
Initialize pos_map ← map of node positions in routes of S
Initialize suffix_demands ← suffix demands cumulative sum for each route of S
Initialize tabu_dict ← empty map (Tabu list)
Initialize penalty_controller (lam)

Initialize S* ← S
Initialize best_dist ← sum(C)
Initialize no_improve ← 0
Initialize iteration ← 0

while iteration < max_iter do
    if no_improve >= max_no_improve then
        break  // Dừng sớm nếu không cải thiện
    end if

    # --- Đánh giá hoặc nhiễu loạn nghiệm (nếu gặp bế tắc) ---
    if no_improve > perturbation_threshold then
        Apply perturbation to S (e.g., relocate random nodes)
        Recompute C, L, pos_map, suffix_demands
        no_improve ← 0
    end if

    best_move ← None
    best_move_cost ← ∞

    # --- Tìm kiếm lân cận Granular Candidate List ---
    for each node u in S (u != 0) do
        for each node v in GranularNeighbors(u) do
            # Xét các nước đi Relocate1, Relocate2, Swap, 2-Opt* liên quan tới u và v
            for each candidate_move in {Relocate(u, v), Swap(u, v), TwoOptStar(u, v)} do
                if candidate_move is infeasible then continue
                
                # Tính chi phí phạt thích nghi O(1) bằng cách dùng cache và hệ số phạt lam
                new_cost ← C[r_u] + C[r_v] + Δ_cost
                new_overload ← max(0, new_load_u - capacity) + max(0, new_load_v - capacity)
                penalized_cost ← new_cost + lam * new_overload

                is_tabu ← tabu_dict.has(candidate_move.key) and tabu_dict[candidate_move.key] > iteration
                aspiration_met ← (new_overload == 0) and (new_cost < best_dist)

                if (not is_tabu or aspiration_met) and (penalized_cost < best_move_cost) then
                    best_move ← candidate_move
                    best_move_cost ← penalized_cost
                end if
            end for
        end for
    end for

    if best_move is None then
        no_improve ← no_improve + 1
        iteration ← iteration + 1
        continue
    end if

    # --- Áp dụng nước đi tốt nhất ---
    Apply best_move to S
    Update C, L, pos_map, suffix_demands incrementally
    Add best_move.key to tabu_dict with tenure

    # --- Cập nhật hệ số phạt thích nghi lam ---
    penalty_controller.register_state(is_feasible(S))
    lam ← penalty_controller.update_penalty(iteration)

    # --- Cập nhật nghiệm tốt nhất ---
    current_dist ← sum(C)
    if is_feasible(S) and current_dist < best_dist then
        S* ← copy(S)
        best_dist ← current_dist
        no_improve ← 0
    else
        no_improve ← no_improve + 1
    end if

    iteration ← iteration + 1
end while

# --- Hậu tối ưu ---
S* ← apply_intra_or_opt(S*)
return S*, route_cost(S*)
```