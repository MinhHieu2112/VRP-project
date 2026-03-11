from pyvrp import Model
from pyvrp.stop import MaxRuntime

class PyVRPSolver:
    def __init__(self, matrix, constraints):
        self.matrix = matrix
        self.constraints = constraints
        self.model = Model()

    def solve(self, time_limit=180, display=True):
        num_points = self.matrix.shape[0]
        nodes = []

        print(f"--- Khởi tạo Model PyVRP cho {num_points} điểm ---")
        
        # 1. Thêm Kho (Depot)
        depot = self.model.add_depot(x=0, y=0)
        nodes.append(depot)

        # 2. Thêm Khách hàng (Clients)
        # Sử dụng default_demand từ JSON nếu có, mặc định là 1
        demand = self.constraints.get('default_demand', 1)
        for _ in range(1, num_points):
            client = self.model.add_client(x=0, y=0, delivery=demand)
            nodes.append(client)

        # 3. Thêm Đội xe (Lấy từ constraints đã parse từ JSON)
        self.model.add_vehicle_type(
            num_available=self.constraints.get('max_vehicles', 200), 
            capacity=self.constraints.get('vehicle_capacity', 10)
        )

        # 4. Nạp Ma trận cạnh (2.56 triệu cạnh)
        # ACVRP: d(i,j) != d(j,i) được đảm bảo vì ta lấy chính xác matrix[i, j]
        print("--- Đang thiết lập ma trận cạnh ---")
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    self.model.add_edge(nodes[i], nodes[j], distance=self.matrix[i, j])

        # 5. Thực hiện giải
        print(f"--- Bắt đầu tối ưu hóa (Limit: {time_limit}s) ---")
        res = self.model.solve(stop=MaxRuntime(time_limit), display=display)
        
        return res