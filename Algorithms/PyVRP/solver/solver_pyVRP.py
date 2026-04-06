import random
from pyvrp import Model
from pyvrp.stop import NoImprovement, MaxIterations, MultipleCriteria


class PyVRPSolver:
    def __init__(self, matrix, constraints):
        self.matrix = matrix
        self.constraints = constraints
        self.model = Model()

    def solve(self, no_improve_iters, display=True):
        """
        Giải ACVRP bằng PyVRP (HGS).

        Stopping criterion: dừng khi không cải thiện sau `no_improve_iters`
        vòng lặp liên tiếp — không giới hạn thời gian cứng.

        Random seed thay đổi mỗi lần chạy để kết quả không bị deterministic.

        Parameters
        ----------
        no_improve_iters : int
            Số vòng lặp không cải thiện tối đa trước khi dừng.
            Mặc định 2000 — tương đương ~60-120s với 1600 điểm.
        display : bool
            Hiển thị log quá trình tối ưu.
        """
        num_points = self.matrix.shape[0]
        nodes = []

        print(f"--- Khởi tạo Model PyVRP cho {num_points} điểm ---")

        # 1. Thêm Kho (Depot)
        depot = self.model.add_depot(x=0, y=0)
        nodes.append(depot)

        # 2. Thêm Khách hàng
        demand = self.constraints.get('default_demand', 1)
        for _ in range(1, num_points):
            client = self.model.add_client(x=0, y=0, delivery=demand)
            nodes.append(client)

        # 3. Thêm Đội xe
        self.model.add_vehicle_type(
            num_available=self.constraints.get('max_vehicles', 200),
            capacity=self.constraints.get('vehicle_capacity', 10)
        )

        # 4. Nạp Ma trận cạnh — ACVRP: d(i,j) != d(j,i)
        print("--- Đang thiết lập ma trận cạnh ---")
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    self.model.add_edge(nodes[i], nodes[j], distance=self.matrix[i, j])

        # 5. Stopping criterion: NoImprovement thay vì MaxRuntime
        #    Dừng khi không cải thiện sau no_improve_iters vòng liên tiếp.
        stop = NoImprovement(max_iterations=no_improve_iters)

        # 6. Random seed khác nhau mỗi lần chạy — tránh kết quả deterministic
        seed = random.randint(0, 2**31 - 1)
        print(f"--- Bắt đầu tối ưu hóa (NoImprovement={no_improve_iters} iters, seed={seed}) ---")

        res = self.model.solve(stop=stop, seed=seed, display=display)

        return res