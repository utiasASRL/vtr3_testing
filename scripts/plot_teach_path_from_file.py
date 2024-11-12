import os
import matplotlib.pyplot as plt
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse

# big_loop
offline_graph_dir = "/home/samqiao/ASRL/vtr3_testing/bash_scripts/radar/20240819/0819_big_loop1/20240819/0819_big_loop1/graph"

online_graph_dir = "/home/samqiao/ASRL/vtr3/temp/0821_big_loop_rosbag/graph"

#small_loop
# offline_graph_dir = "/home/samqiao/ASRL/vtr3_testing/bash_scripts/radar/20240819/0819_small_loop3/20240819/0819_small_loop3/graph"

# online_graph_dir = "/home/samqiao/ASRL/vtr3/temp/0821_small_loop_rosbag/graph"


if __name__ == '__main__':

    factory = Rosbag2GraphFactory(offline_graph_dir)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

    g_utils.set_world_frame(test_graph, test_graph.root)

    v_start = test_graph.root

    path_matrix = vtr_path.path_to_matrix(test_graph, PriviledgedIterator(v_start))
    print(path_matrix.shape)

    x = []
    y = []
    t = []

    for v, e in PriviledgedIterator(v_start):
        x.append(v.T_v_w.r_ba_ina()[0])
        y.append(v.T_v_w.r_ba_ina()[1])
        t.append(v.stamp / 1e9)


    plt.figure(0)
    plt.plot(x, y, label="Teach_offline", linewidth=5)
    plt.axis('equal')
    # plt.hold(True)


    factory = Rosbag2GraphFactory(online_graph_dir)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

    g_utils.set_world_frame(test_graph, test_graph.root)

    v_start = test_graph.root

    path_matrix = vtr_path.path_to_matrix(test_graph, PriviledgedIterator(v_start))
    print(path_matrix.shape)

    x = []
    y = []
    t = []

    for v, e in PriviledgedIterator(v_start):
        x.append(v.T_v_w.r_ba_ina()[0])
        y.append(v.T_v_w.r_ba_ina()[1])
        t.append(v.stamp / 1e9)

    plt.plot(x, y, label="Teach_online", linewidth=5)
    plt.axis('equal')
    plt.legend()

    plt.show()

