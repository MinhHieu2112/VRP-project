from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from .operators.destroy_operators import random_removal, worst_removal
from .operators.repair_operators import greedy_insertion, regret_insertion

# FIX: DataLoader scale raw_meters / 10 → units (1 unit = 10m).
# Để ra km phải chia KM_SCALE = 100, không phải 1000.
# Dùng import từ Pipeline để đảm bảo nhất quán với toàn bộ project.
try:
    from Utils.Pipeline import KM_SCALE
except ImportError:
    KM_SCALE = 100   # fallback: 1 unit = 10m → 100 units = 1 km


def configure_alns(initial_state, config):
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

    def on_best_found(state, rnd_state):
        # FIX: chia KM_SCALE (100) thay vì 1000
        actual_km = sum(
            state.route_cost(r)
            for r in state.routes
            if len(r) > 2
        ) / KM_SCALE
        unassigned = len(state.unassigned)
        print(
            f"[ALNS] Lời giải tốt hơn: {actual_km:.2f} km"
            f" | Chưa gán: {unassigned} node"
        )

    alns.on_best(on_best_found)
    return alns, accept, select, on_best_found