from Utils.data_loader import DataLoader
from Utils.visualizer import Visualizer
from solver_OR_Tools import ORToolsSolver
# from solver_ils import ILSSolver  <-- Import tương tự khi bạn xong file ILS

def main():
    # 1. Load Data
    loader = DataLoader("Data/orsm_matrix.csv", "Data/locations.csv")
    data = loader.load_data(num_vehicles=200, vehicle_capacity=10)

    # 2. Chạy OR-Tools
    ort_solver = ORToolsSolver(data)
    ort_routes, ort_dist = ort_solver.solve(time_limit=300)
    if ort_routes:
        print(f"Result OR-Tools: {ort_dist:.2f} km")
    else :
        print("Không tìm thấy lời giải !")

    # 3. Vẽ bản đồ
    viz = Visualizer(data['df_locations'])
    viz.draw(ort_routes, "map_ortools.html")

if __name__ == "__main__":
    main()

