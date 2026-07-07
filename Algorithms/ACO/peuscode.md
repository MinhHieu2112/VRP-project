## Ant Colony Optimization (ACS variant) — Tối ưu hóa Object Pooling

```pascal
# ─── Khởi tạo ────────────────────────────────────────────────────────
Initialize dist_mat      ← asymmetric distance matrix (N×N)
Initialize heuristic_mat ← 1/dist_mat  (N×N, 0 tại đường chéo)

Build seed solution S_seed using greedy/clarke_wright
seed_cost ← route_cost(S_seed)

tau_0 ← 1.0 / (seed_cost × N)         // mức pheromone khởi điểm
Initialize pheromone_mat ← tau_0       // mảng pheromone (N×N, float32)
Boost pheromone on edges of S_seed × seed_weight  // định hướng kiến sớm

Build candidate_list[i] ← top-K nearest neighbors of i  // dùng argpartition O(N)
Initialize ants[0..M-1]  ← M đối tượng Ant cố định      // Object Pooling

Initialize best_path     ← flat_path(S_seed)
Initialize best_distance ← seed_cost
Initialize no_improve    ← 0
q0_current               ← q0

# ─── Vòng lặp chính ─────────────────────────────────────────────────
for iteration = 1 to max_iter do

    if no_improve >= no_improve_limit then
        break  // dừng sớm
    end if

    if no_improve == half_limit then
        q0_current ← max(q0 - 0.2, 0.5)  // mở rộng khám phá khi bế tắc
    else if no_improve == 0 then
        q0_current ← q0
    end if

    # --- Mỗi kiến xây dựng một lộ trình ---
    for each ant in ants do
        ant.reset()  // O(N), dùng lại đối tượng thay vì tạo mới
        ant ← construct_solution(ant, q0_current, dist_mat,
                                  pheromone_mat, heuristic_mat,
                                  candidate_list)
    end for

    # --- Cập nhật nghiệm tốt nhất ---
    improved ← False
    for each ant in ants do
        if ant.total_distance < best_distance then
            best_path     ← copy(ant.travel_path)
            best_distance ← ant.total_distance
            improved      ← True
        end if
    end for

    no_improve ← 0 if improved else no_improve + 1

    # --- Global pheromone update (chỉ trên best_path) ---
    for each edge (u, v) in best_path do
        pheromone_mat[u][v] ← (1 - rho) × pheromone_mat[u][v]
                               + rho / best_distance
    end for

end for

return best_path, best_distance
```

---

```pascal
# ─── construct_solution(ant, q0, ...) ──────────────────────────────
# Xây dựng lộ trình hoàn chỉnh cho một con kiến dựa trên tri thức pheromone.

Initialize local_update_batch ← []
customer_steps ← 0

while ant.unvisited_count > 0 do

    if customer_steps > N - 1 then
        force_visit_remaining(ant)   // xử lý trường hợp kẹt
        break
    end if

    # Bước 1: Lấy candidate list của node hiện tại
    current ← ant.current_index
    feasible ← []
    for each nb in candidate_list[current] do
        if nb == -1 then break         // padding
        if not ant.visited[nb] and ant.load + demand[nb] <= capacity then
            feasible.append(nb)
        end if
    end for

    # Bước 2: Fallback toàn đồ thị nếu candidate list rỗng
    if feasible is empty then
        for idx = 1 to N-1 do
            if not ant.visited[idx] and ant.load + demand[idx] <= capacity then
                feasible.append(idx)
            end if
        end for
    end if

    # Bước 3: Trả về depot nếu không còn node khả thi
    if feasible is empty then
        local_update_batch.append((current, 0))
        ant.move_to(0)
    else
        next ← select_next_index(ant, feasible, q0,
                                  pheromone_mat, heuristic_mat)
        local_update_batch.append((current, next))
        ant.move_to(next)
        customer_steps += 1
    end if

end while

if ant.current_index != 0 then
    local_update_batch.append((ant.current_index, 0))
    ant.move_to(0)
end if

# Batch local pheromone update sau khi kiến hoàn thành
for each (i, j) in local_update_batch do
    pheromone_mat[i][j] ← (1 - xi) × pheromone_mat[i][j] + xi × tau_0
end for
```

---

```pascal
# ─── select_next_index(ant, feasible, q0, ...) ─────────────────────
# Chọn node tiếp theo theo quy tắc chuyển trạng thái ACS bằng Python thuần.

current ← ant.current_index
scores  ← []
total   ← 0.0

for each node in feasible do
    p ← pheromone_mat[current][node]
    h ← heuristic_mat[current][node]
    score ← (p ^ alpha) × (h ^ beta)
    if score is not finite then score ← 0.0
    scores.append(score)
    total += score
end for

if total <= 0 then
    return random_choice(feasible)
end if

if random() < q0 then
    # Exploitation — chọn node có score cao nhất
    return feasible[argmax(scores)]
else
    # Exploration — Roulette Wheel trực tiếp trên list Python
    r ← random() × total
    cumulative ← 0.0
    for idx, score in enumerate(scores) do
        cumulative += score
        if cumulative >= r then
            return feasible[idx]
        end if
    end for
    return feasible[-1]
end if
```
