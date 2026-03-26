import numpy as np
import pandas as pd
import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Test the fixes
print("Testing ACO fixes...")

# Load data
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Utils', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

locations_file = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['locations_data'])
matrix_file = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['distance_matrix'])

print("Loading data...")
locations_df = pd.read_csv(locations_file)
matrix_df = pd.read_csv(matrix_file, header=None)
node_dist_mat = matrix_df.values

print(f"Matrix shape: {node_dist_mat.shape}")
print(f"Diagonal values: {np.diag(node_dist_mat)}")

# Test heuristic matrix creation
heuristic_mat = np.copy(node_dist_mat)
np.fill_diagonal(heuristic_mat, np.inf)
heuristic_info_mat = 1 / heuristic_mat
np.fill_diagonal(heuristic_info_mat, 0)

print(f"Heuristic matrix diagonal: {np.diag(heuristic_info_mat)}")
print(f"Heuristic matrix has inf: {np.any(np.isinf(heuristic_info_mat))}")
print(f"Heuristic matrix has nan: {np.any(np.isnan(heuristic_info_mat))}")

print("Test completed successfully!")