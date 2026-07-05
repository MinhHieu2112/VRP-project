# File re-export các toán tử áp dụng nước đi từ module tập trung Utils.local_search.
from Utils.Operators.local_search import (
    apply_relocate,
    apply_relocate2,
    apply_swap,
    apply_2opt_star,
)

__all__ = ["apply_relocate", "apply_relocate2", "apply_swap", "apply_2opt_star"]
