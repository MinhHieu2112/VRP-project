# File định nghĩa bộ cấu hình và điều phối các toán tử trong thuật toán ALNS.
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy.random as rnd
from alns import ALNS, State
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel

from .operators.destroy_operators import random_removal, worst_removal
from .operators.repair_operators import greedy_insertion, regret_insertion

if TYPE_CHECKING:
    from .state import CvrpState

try:
    from Utils.Pipeline import KM_SCALE
except ImportError:
    KM_SCALE = 100


def configure_alns(initial_state, config):
    # Cấu hình đối tượng ALNS với các toán tử phá hủy, tái thiết, lựa chọn và chấp nhận.
    alns = ALNS()

    alns.add_destroy_operator(random_removal)
    alns.add_destroy_operator(worst_removal)
    alns.add_repair_operator(greedy_insertion)
    alns.add_repair_operator(regret_insertion)

    params = config["alns_parameters"]

    select = RouletteWheel(
        scores=params["scores"],
        num_destroy=2,
        num_repair=2,
        decay=params["decay"]
    )

    accept = SimulatedAnnealing(
        start_temperature=params["start_temperature"],
        end_temperature=params["end_temperature"],
        step=params["step"],
        method="exponential"
    )

    def on_best_found(state: State, rng: rnd.Generator, **kwargs) -> None:
        # Gọi lại khi thuật toán ALNS tìm thấy phương án tối ưu tốt nhất mới.
        s = cast("CvrpState", state)
        actual_km = sum(
            s.route_cost(r) for r in s.routes if len(r) > 2
        ) / KM_SCALE
        print(
            f"[ALNS] Lời giải tốt hơn: {actual_km:.2f} km"
            f" | Chưa gán: {len(s.unassigned)} node"
        )

    alns.on_best(on_best_found)
    return alns, accept, select, on_best_found