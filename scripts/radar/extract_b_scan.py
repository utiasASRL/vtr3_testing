import os
import matplotlib.pyplot as plt
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory

from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse

# point cloud vis
from sensor_msgs_py.point_cloud2 import read_points
# import open3d as o3d
from pylgmath import Transformation
from vtr_utils.plot_utils import convert_points_to_frame, extract_map_from_vertex, downsample, extract_points_from_vertex, range_crop
import time

import yaml
import cv2

# print current working directory
print("Current working dir", os.getcwd())

# Insert path at index 0 so it's searched first
import sys
sys.path.insert(0, "scripts")

from scripts.radar.utils.helper import *

parent_folder = "/home/samqiao/ASRL/vtr3_testing"

def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file.

    :param config_path: Path to the YAML configuration file.
    :return: A dictionary representing the configuration.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    config = load_config(os.path.join(parent_folder,'scripts/config.yaml'))

    # Access database configuration
    db = config['radar_data']
    db_loop = db.get('woody')
    db_rosbag_path = db_loop.get('rosbag_path')

    teach_rosbag_path = db_rosbag_path.get('teach')
    # repeat_rosbag_path = db_rosbag_path.get('repeat1') # dont think this is needed

    # for pose graph
    pose_graph_path = db_loop.get('pose_graph_path').get('woody')
    print("pose graph path:",pose_graph_path)

    db_bool = config['bool']
    SAVE = db_bool.get('SAVE')
    # print("SAVE:",SAVE)
    PLOT = db_bool.get('PLOT')

    result_folder = config.get('output')

    save_folder = os.path.join(result_folder, "ICRA_woody_teach")


    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Folder '{save_folder}' created.")

    outpath = os.path.join(save_folder,f"parking_teach_radar_association_teach.npz")

    SAVE = True

    # print("I am in the if clause")
    fft_data,radar_timestamps,azimuth_angles, azimuth_timestamps_total,cart_imgs = get_radar_scan_images_and_timestamps(teach_rosbag_path)
    print("fft_data shape:", np.array(fft_data).shape)
    print("radar_timestamps shape:", np.array(radar_timestamps).shape)
    print("azimuth_angles shape:", np.array(azimuth_angles).shape)
    print("azimuth_timestamps_total shape:", np.array(azimuth_timestamps_total).shape)


    scan_folder = os.path.join(save_folder, "radar_scans")

    for timestamp in radar_timestamps:
        print("processing scan index:",radar_timestamps.index(timestamp))
        print("radar scan timestamp:",timestamp)
        polar_img = fft_data[radar_timestamps.index(timestamp)]

        encoder_values = np.array(azimuth_angles[radar_timestamps.index(timestamp)])
        # print("encoder_values:",encoder_values)

        azimuth_timestamps = np.array(azimuth_timestamps_total[radar_timestamps.index(timestamp)])
        # print("azimuth_timestamps:",azimuth_timestamps)

        combined = np.vstack((azimuth_timestamps,encoder_values)).T
        print("combined shape:",combined.shape)

        if not os.path.exists(scan_folder):
            os.makedirs(scan_folder)
            print(f"Folder '{scan_folder}' created.")

        cv2.imwrite(os.path.join(scan_folder,f"{timestamp}.png"), polar_img)
        cv2.imshow("polar_img",cart_imgs[radar_timestamps.index(timestamp)])
        np.savetxt(os.path.join(scan_folder,f"{timestamp}.txt"), combined, delimiter=",", fmt='%s')

        
        





    