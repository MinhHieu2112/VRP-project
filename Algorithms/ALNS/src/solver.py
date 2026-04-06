from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from .operators.destroy_operators import random_removal, worst_removal
from .operators.repair_operators import greedy_insertion, regret_insertion

METERS_TO_KM = 1000


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
        # FIX: chia 1000 (mét → km) trực tiếp, không dùng scaling_factor từ config
        actual_km = sum(
            state.route_cost(r)
            for r in state.routes
            if len(r) > 2
        ) / METERS_TO_KM
        unassigned = len(state.unassigned)
        print(
            f"[ALNS] Lời giải tốt hơn: {actual_km:.2f} km"
            f" | Chưa gán: {unassigned} node"
        )

    alns.on_best(on_best_found)
    return alns, accept, select, on_best_found
