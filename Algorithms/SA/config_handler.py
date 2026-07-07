# File chịu trách nhiệm định nghĩa, phân tích và tải cấu hình cho thuật toán SA.
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class SAConfig:
    """Cấu trúc dữ liệu chứa tất cả cấu hình và tham số vận hành của thuật toán SA."""
    capacity:        float = 10.0
    demand:          float = 1.0
    max_v:           int   = 200

    T_start:         float = 5000.0
    T_min:           float = 0.1
    alpha:           float = 0.9997
    max_no_improve:  int   = 1000
    iter_per_T:      int   = 500

    init_strategy:   str   = "clarke_wright"

    vehicle_penalty: int   = 3000

    _valid_strategies: tuple = field(
        default=("random", "greedy", "clarke_wright"),
        init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        # Thực hiện kiểm tra tính hợp lệ của cấu hình sau khi khởi tạo.
        self._validate()

    def _validate(self) -> None:
        # Kiểm tra chi tiết tính hợp lệ của các tham số cấu hình.
        if self.T_start <= self.T_min:
            raise ValueError(
                f"[SAConfig] start_temperature ({self.T_start}) phải > "
                f"end_temperature ({self.T_min})"
            )
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(
                f"[SAConfig] alpha (cooling rate) phải trong (0, 1), "
                f"nhận được {self.alpha}"
            )
        if self.capacity <= 0:
            raise ValueError(
                f"[SAConfig] vehicle_capacity phải > 0, nhận được {self.capacity}"
            )
        if self.init_strategy not in self._valid_strategies:
            raise ValueError(
                f"[SAConfig] init_strategy '{self.init_strategy}' không hợp lệ. "
                f"Chọn trong: {self._valid_strategies}"
            )


def load_sa_config(config: dict) -> SAConfig:
    # Đọc thông tin cấu hình từ một dictionary cấu hình chung.
    cons = config.get("global_constraints", config.get("constraints", {}))
    sa_cfg = config.get("sa_parameters", config.get("alns_parameters", {}))

    return SAConfig(
        capacity        = float(cons.get("vehicle_capacity", 10)),
        demand          = float(cons.get("default_demand",    1)),
        max_v           = int(cons.get("max_vehicles",       200)),
        T_start         = float(sa_cfg.get("start_temperature", 5000)),
        T_min           = float(sa_cfg.get("end_temperature",    0.1)),
        alpha           = float(sa_cfg.get("step",             0.9997)),
        max_no_improve  = int(sa_cfg.get("max_no_improve",    1000)),
        iter_per_T      = int(sa_cfg.get("iter_per_temp",      500)),
        init_strategy   = str(sa_cfg.get("init_strategy", "clarke_wright")),
        vehicle_penalty = int(sa_cfg.get("vehicle_penalty", cons.get("vehicle_penalty", 3000))),
    )


def load_sa_config_from_file(file_path: str | None = None) -> SAConfig:
    # Tải cấu hình thuật toán SA trực tiếp từ tệp config.json.
    import os
    import json
    if file_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "config.json")
    with open(file_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    return load_sa_config(config_dict)
