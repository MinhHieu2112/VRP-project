# Định nghĩa các cấu trúc dữ liệu mô tả dịch chuyển lân cận (Move Types) của GTS.
from __future__ import annotations
from typing import List, NamedTuple, Union

Route = List[int]
Solution = List[Route]


class MoveRel1(NamedTuple):
    """Đại diện cho dịch chuyển di chuyển 1 khách hàng sang vị trí mới."""
    u:     int
    r_src: int
    p_u:   int
    r_dst: int
    p_ins: int


class MoveRel2(NamedTuple):
    """Đại diện cho dịch chuyển di chuyển 2 khách hàng liên tiếp sang vị trí mới."""
    u:     int
    v_nxt: int
    r_src: int
    p_u:   int
    r_dst: int
    p_ins: int


class MoveSwap(NamedTuple):
    """Đại diện cho dịch chuyển đổi chỗ vị trí của hai khách hàng."""
    u:   int
    v:   int
    r_u: int
    p_u: int
    r_v: int
    p_v: int


class Move2OptStar(NamedTuple):
    """Đại diện cho dịch chuyển 2-opt* chéo tuyến."""
    r1: int
    i:  int
    r2: int
    j:  int


AnyMove = Union[MoveRel1, MoveRel2, MoveSwap, Move2OptStar]
