import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import pandas as pd
    import numpy as np
    print("Basic imports OK")
except ImportError as e:
    print(f"Import error: {e}")

try:
    from cvrp_base import CVRPGraph, Node
    from basic_aco import BasicACO
    print("ACO imports OK")
except ImportError as e:
    print(f"ACO import error: {e}")

try:
    from Utils.ResultHandler import ResultHandler
    from Utils.Visualizer import Visualizer
    print("Utils imports OK")
except ImportError as e:
    print(f"Utils import error: {e}")

print("Test completed")