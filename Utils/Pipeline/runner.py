# Module định nghĩa lớp AlgorithmRunner chuẩn hóa quy trình chạy thuật toán VRP (Template Method pattern).
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from .Pipeline import load_data, build_result, save_result, visualize


class AlgorithmRunner(ABC):
    """Lớp cơ sở chuẩn hóa quy trình nạp cấu hình, giải bài toán và lưu kết quả cho mọi thuật toán VRP."""

    def __init__(
        self,
        name:        str,
        config_path: str,
        subfolder:   Optional[str] = None,
    ) -> None:
        # Khởi tạo tên thuật toán, đường dẫn config và thư mục lưu kết quả.
        self.name        = name
        self.config_path = config_path
        self.subfolder   = subfolder or name

    def _load_config(self) -> Dict[str, Any]:
        # Đọc và trả về cấu hình từ tệp JSON tại đường dẫn đã đăng ký.
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"[{self.name}] Không tìm thấy config: {self.config_path}"
            )
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @abstractmethod
    def build_solver(self, data: Dict[str, Any], config: Dict[str, Any]) -> Any:
        # Phương thức trừu tượng xây dựng và trả về đối tượng solver đặc thù của mỗi thuật toán.
        ...

    def _print_summary(self, result: Dict[str, Any], elapsed: float) -> None:
        # In tóm tắt kết quả gồm quãng đường, số xe và thời gian thực thi.
        print(
            f"\n[{self.name.upper()} DONE] "
            f"{result['total_distance_km']:.2f} km | "
            f"{result['num_vehicles']} xe | "
            f"{elapsed:.2f}s"
        )

    def run(self) -> Dict[str, Any]:
        # Thực hiện toàn bộ pipeline: nạp dữ liệu -> giải -> đo thời gian -> lưu kết quả -> vẽ bản đồ.
        config = self._load_config()
        data   = load_data(config)

        solver = self.build_solver(data, config)

        t0 = time.time()
        routes, cost = solver.solve()
        elapsed = time.time() - t0

        result = build_result(self.name, routes, cost, elapsed)
        save_result(result, config, self.subfolder)
        visualize(result, config, self.subfolder, data["df_locations"])

        self._print_summary(result, elapsed)
        return result
