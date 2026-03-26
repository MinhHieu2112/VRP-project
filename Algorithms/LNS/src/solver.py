from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RandomSelect  # Đổi từ RouletteWheel sang RandomSelect
from .operators.destroy_operators import random_removal, worst_removal, shaw_removal
from .operators.repair_operators import greedy_insertion, regret_insertion

def configure_lns(initial_state, config):
    alns = ALNS()

    # Đăng ký toán tử (nên dùng các toán tử mạnh cho LNS)
    alns.add_destroy_operator(shaw_removal)
    alns.add_repair_operator(regret_insertion)

    # Lấy đúng nhóm lns_parameters (hoặc alns_parameters tùy file config)
    params = config.get('lns_parameters', config.get('alns_parameters'))
    
    # TRUY CẬP ĐÚNG CẤU TRÚC LỒNG NHAU
    sa_params = params['simulated_annealing']
    
    select = RandomSelect(num_destroy=1, num_repair=1)

    accept = SimulatedAnnealing(
        start_temperature=sa_params['start_temperature'], 
        end_temperature=sa_params['end_temperature'], 
        step=sa_params['step'],
        method="exponential"
    )

    def on_best_found(state, rnd_state):
        # Chia 100 vì ma trận của bạn đang để đơn vị là 10m (scaled)
        print(f"[LNS] Tìm thấy lời giải tốt nhất mới: {state.objective() / 100:.2f} km")
    
    return alns, accept, select, on_best_found