import pandas as pd
import numpy as np
import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Test data loading
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Utils', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

locations_file = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['locations_data'])
matrix_file = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['distance_matrix'])

print("Loading locations...")
locations_df = pd.read_csv(locations_file)
print(f"Locations shape: {locations_df.shape}")

print("Loading matrix...")
matrix_df = pd.read_csv(matrix_file, header=None)
print(f"Matrix shape: {matrix_df.shape}")

print("Data loaded successfully!")