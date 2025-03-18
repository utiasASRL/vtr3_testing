import os
import time

import numpy as np
import matplotlib.pyplot as plt

from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import argparse


from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator, SpatialIterator
import vtr_pose_graph.graph_utils as g_utils



if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog = 'Plot Point Clouds Path',
                            description = 'Plots point clouds')
    parser.add_argument('-g', '--graph', default=os.getenv("VTRDATA"), help="The filepath to the pose graph folder. (Usually /a/path/graph)")      # option that takes a value
    parser.add_argument('-r', '--run', type=int, help="Select the run")
    parser.add_argument('-v', '--vertex', type=int, help="Select the vertex")
    parser.add_argument('-m', '--message', help="The msg stored in the vertex to load.")
    args = parser.parse_args()

    factory = Rosbag2GraphFactory(args.graph)
    bridge = CvBridge()


    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

    g_utils.set_world_frame(test_graph, test_graph.root)
    print((args.run, args.vertex))
    vertex = test_graph.get_vertex((args.run, args.vertex))

    raw_msg = vertex.get_data(args.message)
    plt.imshow(bridge.imgmsg_to_cv2(raw_msg.b_scan_img))
    plt.show()