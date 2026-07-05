# Lớp bọc thực thi thuật toán tối ưu hóa VRP bằng thư viện PyVRP (Hybrid Genetic Search).
import random
from pyvrp import Model
from pyvrp.stop import NoImprovement


class PyVRPSolver:
    """Bộ giải VRP sử dụng thuật toán Hybrid Genetic Search của thư viện PyVRP."""

    def __init__(self, matrix, constraints):
        # Khởi tạo mô hình PyVRP với ma trận khoảng cách và các ràng buộc.
        self.matrix = matrix
        self.constraints = constraints
        self.model = Model()

    def solve(self, no_improve_iters, display=True):
        # Xây dựng mô hình và thực thi tối ưu hóa VRP bằng PyVRP.
        num_points = self.matrix.shape[0]
        nodes = []

        print(f"--- Khởi tạo Model PyVRP cho {num_points} điểm ---")

        depot = self.model.add_depot(x=0, y=0)
        nodes.append(depot)

        demand = self.constraints.get('default_demand', 1)
        for _ in range(1, num_points):
            client = self.model.add_client(x=0, y=0, delivery=demand)
            nodes.append(client)

        self.model.add_vehicle_type(
            num_available=self.constraints.get('max_vehicles', 200),
            capacity=self.constraints.get('vehicle_capacity', 10)
        )

        print("--- Đang thiết lập ma trận cạnh ---")
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    self.model.add_edge(nodes[i], nodes[j], distance=self.matrix[i, j])

        stop = NoImprovement(max_iterations=no_improve_iters)

        seed = random.randint(0, 2**31 - 1)
        print(f"--- Bắt đầu tối ưu hóa (NoImprovement={no_improve_iters} iters, seed={seed}) ---")

        res = self.model.solve(stop=stop, seed=seed, display=display)

        return res