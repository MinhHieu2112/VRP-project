import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Test the route parsing logic
best_path = [0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 0]  # Example path
best_distance = 958.86
best_vehicles = 3

print("Testing route parsing...")

# Parse best_path into routes dictionary
routes_dict = {}
current_route = [0]  # Start with depot
vehicle_id = 0

for i in range(1, len(best_path)):  # Skip the first 0
    node = best_path[i]
    current_route.append(node)
    if node == 0:  # Complete route when returning to depot
        routes_dict[vehicle_id] = current_route.copy()
        vehicle_id += 1
        current_route = [0]  # Start new route from depot

# Handle the last route if it doesn't end with 0
if len(current_route) > 1:
    current_route.append(0)
    routes_dict[vehicle_id] = current_route

print(f"Parsed routes: {routes_dict}")

# Create standardized result
standardized_result = {
    "solver_name": "ACO",
    "total_distance_km": best_distance,
    "execution_time": 0.0,
    "routes": routes_dict,
    "num_vehicles": best_vehicles
}

print(f"Standardized result: {standardized_result}")

print("Test completed successfully!")