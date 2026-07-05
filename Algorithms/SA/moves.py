# File định nghĩa các phép thử thay đổi cấu trúc nghiệm (swap, relocate, intra_swap) phục vụ SA.
from __future__ import annotations

import random
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

from Utils.Operators.local_search import route_load

Route = List[int]


class MoveResult(NamedTuple):
    """Cấu trúc dữ liệu đại diện cho kết quả thực hiện một dịch chuyển nghiệm."""
    accepted:  bool
    old_costs: Tuple[float, ...]
    rollback:  Optional[Callable[[], None]]


def try_swap(
    r1:          Route,
    r2:          Route,
    idx1:        int,
    idx2:        int,
    demands_map: Dict[int, float],
    capacity:    float,
    cost_r1:     float,
    cost_r2:     float,
) -> MoveResult:
    # Thử đổi chỗ ngẫu nhiên hai khách hàng giữa hai tuyến đường khác nhau.
    node1 = r1[idx1]
    node2 = r2[idx2]
    d1    = demands_map.get(node1, 0.0)
    d2    = demands_map.get(node2, 0.0)

    load_r1 = route_load(demands_map, r1)
    load_r2 = route_load(demands_map, r2)

    new_load_r1 = load_r1 - d1 + d2
    new_load_r2 = load_r2 - d2 + d1

    if new_load_r1 > capacity or new_load_r2 > capacity:
        return MoveResult(accepted=False, old_costs=(), rollback=None)

    r1[idx1], r2[idx2] = r2[idx2], r1[idx1]

    def _rollback() -> None:
        r1[idx1], r2[idx2] = r2[idx2], r1[idx1]

    return MoveResult(
        accepted=True,
        old_costs=(cost_r1, cost_r2),
        rollback=_rollback,
    )


def try_relocate(
    r1:          Route,
    r2:          Route,
    idx1:        int,
    demands_map: Dict[int, float],
    capacity:    float,
    cost_r1:     float,
    cost_r2:     float,
) -> MoveResult:
    # Thử chuyển một khách hàng từ tuyến đường này sang tuyến đường kia.
    node   = r1[idx1]
    d_node = demands_map.get(node, 0.0)
    load_r2 = route_load(demands_map, r2)

    if load_r2 + d_node > capacity:
        return MoveResult(accepted=False, old_costs=(), rollback=None)

    ins_pos = random.randint(1, len(r2) - 1)

    r1.pop(idx1)
    r2.insert(ins_pos, node)

    def _rollback() -> None:
        r2.pop(ins_pos)
        r1.insert(idx1, node)

    return MoveResult(
        accepted=True,
        old_costs=(cost_r1, cost_r2),
        rollback=_rollback,
    )


def try_intra_swap(
    r1:      Route,
    cost_r1: float,
) -> MoveResult:
    # Thử đổi chỗ hai khách hàng trong nội bộ cùng một tuyến đường.
    if len(r1) < 4:
        return MoveResult(accepted=False, old_costs=(), rollback=None)

    i, j = random.sample(range(1, len(r1) - 1), 2)

    r1[i], r1[j] = r1[j], r1[i]

    def _rollback() -> None:
        r1[i], r1[j] = r1[j], r1[i]

    return MoveResult(
        accepted=True,
        old_costs=(cost_r1,),
        rollback=_rollback,
    )
