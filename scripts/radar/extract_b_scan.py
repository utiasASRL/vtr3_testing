import os
import matplotlib.pyplot as plt
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory

from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse
from scripts.radar.utils.helper import *


# point cloud vis
from sensor_msgs_py.point_cloud2 import read_points
# import open3d as o3d
from pylgmath import Transformation
from vtr_utils.plot_utils import convert_points_to_frame, extract_map_from_vertex, downsample, extract_points_from_vertex, range_crop
import time

# NOT finished 

# print the current working directory
print("Current working dir", os.getcwd())

pose_graph_path = "../../posegraph/20241114/k_strong1/graph"


factory = Rosbag2GraphFactory(pose_graph_path)


test_graph = factory.buildGraph()
print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

g_utils.set_world_frame(test_graph, test_graph.root)

# I will use the repeat path for now
v_start = test_graph.get_vertex((0, 0))

frame = 0

for vertex, e in TemporalIterator(v_start):
    pass