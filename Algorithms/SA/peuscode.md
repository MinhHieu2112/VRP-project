# Cẩm nang Pseudocode giải thuật VRP Suite

Tài liệu này cung cấp pseudocode chi tiết cho ba giải thuật cốt lõi trong hệ thống VRP: **Simulated Annealing (SA)**, **Granular Tabu Search (GTS)**, và **Adaptive Large Neighborhood Search (ALNS)**.

---

## Simulated Annealing (SA) - Tối ưu hóa Delta Evaluation $O(1)$

```pascal
Initialize S using initial strategy (e.g., greedy/clarke_wright)
Initialize C ← array of individual route costs for each route in S
Initialize L ← array of individual route loads for each route in S
Initialize current_cost ← sum(C) + len(S) * vehicle_penalty

Initialize S* ← S
Initialize best_cost ← current_cost
Initialize T ← T_start

while T > T_min do
    improved_this_temp = False

    for iter = 1 to iter_per_T do
        Select two distinct routes r1, r2 randomly from S with indices idx1, idx2
        if length(r1) <= 2 then continue

        Select move_type randomly from [0, 1]

        # --- Phép thử 1: Inter-Route Swap (Tráo đổi 2 khách hàng giữa 2 xe) ---
        if move_type < 0.4 and length(r2) > 2 then
            Select node u at index i from r1, node v at index j from r2 randomly
            
            new_load1 = L[idx1] - demand(u) + demand(v)
            new_load2 = L[idx2] - demand(v) + demand(u)

            if new_load1 <= capacity and new_load2 <= capacity then
                Δ_r1, Δ_r2 = eval_swap_delta(dist, r1, r2, i, j)  // Tính delta cost ở O(1)
                Δ = Δ_r1 + Δ_r2

                if Δ < 0 or random() < exp(-Δ / T) then
                    Swap u and v in S: r1[i] ↔ r2[j]
                    L[idx1] = new_load1, L[idx2] = new_load2
                    C[idx1] = C[idx1] + Δ_r1, C[idx2] = C[idx2] + Δ_r2
                    current_cost = current_cost + Δ
                    
                    if current_cost < best_cost then
                        S* ← copy(S), best_cost ← current_cost, improved_this_temp ← True
                    end if
                end if
            end if

        # --- Phép thử 2: Inter-Route Relocate (Chuyển 1 khách hàng sang xe khác) ---
        else if move_type < 0.8 then
            Select node u at index i from r1 randomly
            new_load2 = L[idx2] + demand(u)

            if new_load2 <= capacity then
                Select insert position ins_pos in r2 randomly
                Δ_r1, Δ_r2 = eval_relocate_delta(dist, r1, r2, i, ins_pos)  // Tính delta cost ở O(1)
                
                Δ = Δ_r1 + Δ_r2
                if length(r1) == 3 then
                    Δ = Δ - vehicle_penalty  // Tối ưu giảm bớt 1 xe rỗng
                end if

                if Δ < 0 or random() < exp(-Δ / T) then
                    Pop u from r1, Insert u into r2 at ins_pos in S
                    L[idx1] = L[idx1] - demand(u)
                    L[idx2] = new_load2
                    C[idx1] = C[idx1] + Δ_r1, C[idx2] = C[idx2] + Δ_r2
                    current_cost = current_cost + Δ
                    
                    if length(r1) <= 2 then
                        Remove empty route r1 from S, L, and C
                    end if
                    
                    if current_cost < best_cost then
                        S* ← copy(S), best_cost ← current_cost, improved_this_temp ← True
                    end if
                end if
            end if

        # --- Phép thử 3: Intra-Route Swap (Đổi chỗ nội bộ 1 xe) ---
        else
            if length(r1) >= 4 then
                Select two indices i, j randomly from r1 (i < j)
                Δ = eval_intra_swap_delta(dist, r1, i, j)  // Tính delta cost ở O(1)

                if Δ < 0 or random() < exp(-Δ / T) then
                    Swap r1[i] and r1[j] in S
                    C[idx1] = C[idx1] + Δ
                    current_cost = current_cost + Δ
                    
                    if current_cost < best_cost then
                        S* ← copy(S), best_cost ← current_cost, improved_this_temp ← True
                    end if
                end if
            end if
        end if
    end for

    T = T * alpha
end while

return S*
```