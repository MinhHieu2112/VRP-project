# File này đã được deprecated.
# Sử dụng Utils/Visualizer.py chung cho toàn bộ project.
# Import redirect để tương thích ngược:
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from Utils.Visualizer import Visualizer
