import os
import matplotlib.pyplot as plt
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_utils.plot_utils import plot_graph
import vtr_pose_graph.graph_utils as g_utils
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Show pose graph',
                        description = 'Plots the connections between the nodes in the pose graph')
    parser.add_argument('-g', '--graph', default=os.getenv("VTRDATA"), help="The filepath to the pose graph folder. (Usually /a/path/graph)")
    parser.add_argument('-f', '--filter', type=int, nargs="*", help="Select only some of the repeat runs. Default plots all runs.")
    args = parser.parse_args()

    offline_graph_dir = args.graph
    factory = Rosbag2GraphFactory(offline_graph_dir)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

    g_utils.set_world_frame(test_graph, test_graph.root)
    plot_graph(test_graph, args.filter)

    
    plt.show()