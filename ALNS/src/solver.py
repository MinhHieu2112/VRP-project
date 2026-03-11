from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from .operators.destroy_operators import random_removal, worst_removal
from .operators.repair_operators import greedy_insertion, regret_insertion

def configure_alns(initial_state, config):
    alns = ALNS()

    alns.add_destroy_operator(random_removal)
    alns.add_destroy_operator(worst_removal)
    alns.add_repair_operator(greedy_insertion)
    alns.add_repair_operator(regret_insertion)

    params = config['alns_parameters']
    
    select = RouletteWheel(
        scores=params['scores'], 
        num_destroy=2, 
        num_repair=2, 
        decay=params['decay']
    )

    accept = SimulatedAnnealing(
        start_temperature=params['start_temperature'], 
        end_temperature=params['end_temperature'], 
        step=params['step'],
        method="exponential"
    )

    # Định nghĩa hàm callback để in tiến trình
    def on_best_found(state, rnd_state):
        print(f"[ALNS] Tìm thấy lời giải tốt hơn: {state.objective() / 100:.2f} km")
    
    # Trả thêm hàm on_best_found về cho main
    return alns, accept, select, on_best_found
    