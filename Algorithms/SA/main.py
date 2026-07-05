# File khởi chạy chính cho thuật toán Simulated Annealing sử dụng AlgorithmRunner chuẩn hóa.
from __future__ import annotations

import os
import sys

_THIS_DIR    = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Utils.Pipeline import AlgorithmRunner
from Algorithms.SA.solver import SimulatedAnnealingSolver


class SARunner(AlgorithmRunner):
    """Runner đặc thù cho Simulated Annealing, chỉ cần định nghĩa cách xây dựng solver."""

    def build_solver(self, data, config):
        # Khởi tạo solver SA từ dữ liệu và cấu hình được nạp sẵn bởi AlgorithmRunner.
        return SimulatedAnnealingSolver(data, config)


if __name__ == "__main__":
    runner = SARunner(
        name        = "SA",
        config_path = os.path.join(_THIS_DIR, "config.json"),
    )
    runner.run()