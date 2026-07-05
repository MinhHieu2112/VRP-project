"""
Utils/Pipeline.py
FIXES:
  [FIX-4] realpath(__file__) thay abspath để PROJECT_ROOT luôn đúng.
"""

import os
from .Data_loader import DataLoader
from .ResultHandler import ResultHandler
from .Visualizer import Visualizer

_THIS_FILE   = os.path.realpath(__file__)
_PIPELINE_DIR = os.path.dirname(_THIS_FILE)
_UTILS_DIR   = os.path.dirname(_PIPELINE_DIR)
PROJECT_ROOT = os.path.dirname(_UTILS_DIR)
KM_SCALE     = DataLoader.KM_SCALE


def load_data(config: dict) -> dict:
    return DataLoader(config).load_data()

def matrix_units_to_km(cost: float) -> float:
    return cost / KM_SCALE

def build_result(solver_name: str, routes, total_cost_units: float, elapsed: float) -> dict:
    route_iter    = routes.values() if isinstance(routes, dict) else routes
    active_routes = {idx: r for idx, r in enumerate(r for r in route_iter if len(r) > 2)}
    return {
        "solver_name":       solver_name,
        "total_distance_km": matrix_units_to_km(total_cost_units),
        "execution_time":    elapsed,
        "routes":            active_routes,
        "num_vehicles":      len(active_routes),
    }

def save_result(result: dict, config: dict, subfolder: str) -> str:
    base_rel   = config.get('paths', {}).get('output_dir', 'Results')
    output_dir = os.path.join(PROJECT_ROOT, base_rel, subfolder)
    return ResultHandler.save_to_txt(result, output_dir)

def visualize(result: dict, config: dict, subfolder: str, df_locations):
    base_rel   = config.get('paths', {}).get('output_dir', 'Results')
    output_dir = os.path.join(PROJECT_ROOT, base_rel, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    vis_cfg  = config.get('visualization', {})
    map_path = os.path.join(output_dir, vis_cfg.get('map_filename', 'route_map.html'))
    try:
        vis = Visualizer(df_locations, osrm_url=vis_cfg.get('osrm_url', 'http://localhost:5001'),
                         use_osrm=vis_cfg.get('use_osrm', True))
        vis.draw(result['routes'], map_path)
        print(f"[Visualizer] Bản đồ: {map_path}")
    except Exception as exc:
        print(f"[Visualizer] Thất bại: {exc}")