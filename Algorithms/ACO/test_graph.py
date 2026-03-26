import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Utils.ResultHandler import ResultHandler
from Utils.Visualizer import Visualizer

print("Testing data loading and graph creation...")

def load_data(locations_file, matrix_file):
    print(f"Loading data from {locations_file} and {matrix_file}")
    locations_df = pd.read_csv(locations_file)
    matrix_df = pd.read_csv(matrix_file, header=None)

    nodes = []
    for idx, row in locations_df.iterrows():
        node = Node(id=int(row['id']), x=float(row['lat']), y=float(row['lon']), demand=1)
        nodes.append(node)

    node_dist_mat = matrix_df.values
    print(f"Loaded {len(nodes)} nodes and distance matrix of shape {node_dist_mat.shape}")
    return len(nodes), nodes, node_dist_mat

# Load config
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Utils', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

locations_file = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['locations_data'])
matrix_file = os.path.join(os.path.dirname(__file__), '..', '..', config['paths']['distance_matrix'])
vehicle_capacity = config['global_constraints']['vehicle_capacity']

print(f"Config loaded. Vehicle capacity: {vehicle_capacity}")

# Load data
node_num, nodes, node_dist_mat = load_data(locations_file, matrix_file)

print("Checking distance matrix for zeros and inf...")
zeros = np.sum(node_dist_mat == 0)
infs = np.sum(np.isinf(node_dist_mat))
nans = np.sum(np.isnan(node_dist_mat))
print(f"Zeros: {zeros}, Infs: {infs}, NaNs: {nans}")

print("Creating graph...")
from cvrp_base import CVRPGraph, Node
import json

graph = CVRPGraph(node_num, nodes, node_dist_mat, vehicle_capacity)

print("Graph created successfully!")
print(f"Pheromone matrix shape: {graph.pheromone_mat.shape}")
print(f"Heuristic matrix shape: {graph.heuristic_info_mat.shape}")
print(f"Heuristic matrix has inf: {np.sum(np.isinf(graph.heuristic_info_mat))}")
print(f"Heuristic matrix has nan: {np.sum(np.isnan(graph.heuristic_info_mat))}")
print("Test completed!")