import os
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

# from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
# from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator
# import vtr_pose_graph.graph_utils as g_utils
# import vtr_regression_testing.path_comparison as vtr_path
# import argparse

import sys
parent_folder = "/home/samqiao/ASRL/vtr3_testing"

# Insert path at index 0 so it's searched first
sys.path.insert(0, parent_folder)

from deps.path_tracking_error.fcns import *

# from radar.utils.helper import *

# # point cloud vis
# from sensor_msgs_py.point_cloud2 import read_points
# # import open3d as o3d
from pylgmath import Transformation
# from vtr_utils.plot_utils import *
# import time

import yaml
import torch
import torchvision

# from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images

# from process_vtr import get_vtr_ptr_baseline

from utils import *
from scripts.visualization.plotter import Plotter


# print("Current working dir", os.getcwd())

T_novatel_robot =  Transformation(T_ba = np.array([[1.000, 0.000, 0.000, 0.550],
  [0.000, 1.000 , 0.000, 0.000],
  [0.000 ,0.000, 1.000 , -1.057],
  [0.000 , 0.000 ,0.000, 1.000]]))

T_radar_robot =  Transformation(T_ba = np.array([[1.000, 0.000, 0.000, 0.025],
                                                 [0.000, -1.000 , 0.000, -0.002],
                                                 [0.000 ,0.000, -1.000 , 1.032],
                                                 [0.000 , 0.000 ,0.000, 1.000]]))



def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file.

    :param config_path: Path to the YAML configuration file.
    :return: A dictionary representing the configuration.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

config = load_config(os.path.join(parent_folder,'scripts/direct/direct_configs/direct_config_sam.yaml'))


db_bool = config['bool']
SAVE = db_bool.get('SAVE')
SAVE = False
print("SAVE:",SAVE)
PLOT = db_bool.get('PLOT')
DEBUG = db_bool.get('DEBUG')

result_folder = config.get('output')


sequence = "grassy_t2_r3"

# change here
out_path_folder = os.path.join(result_folder,sequence)
if not os.path.exists(out_path_folder):
    os.makedirs(out_path_folder)
    print(f"Folder '{out_path_folder}' created.")
else:
    print(f"Folder '{out_path_folder}' already exists.")    


sequence_path = os.path.join(result_folder, sequence)
if not os.path.exists(sequence_path):
    print("ERROR: No sequence found in " + sequence_path)
    exit(0)


TEACH_FOLDER = os.path.join(sequence_path, "teach")
REPEAT_FOLDER = os.path.join(sequence_path, "repeat")
RESULT_FOLDER = os.path.join(sequence_path, "direct")

if not os.path.exists(TEACH_FOLDER):
    raise FileNotFoundError(f"Teach folder {TEACH_FOLDER} does not exist.")
if not os.path.exists(REPEAT_FOLDER):
    raise FileNotFoundError(f"Repeat folder {REPEAT_FOLDER} does not exist.")
if not os.path.exists(RESULT_FOLDER):
    raise FileNotFoundError(f"Result folder {RESULT_FOLDER} does not exist.")

teach_df = np.load(os.path.join(TEACH_FOLDER, "teach.npz"),allow_pickle=True)

# in the teach
# 1. (932,400,1712) images
teach_polar_imgs = teach_df['teach_polar_imgs']
# 2. (932,400, 1) azimuth angles
teach_azimuth_angles = teach_df['teach_azimuth_angles']
# 3. (932,400, 1) azimuth timestamps
teach_azimuth_timestamps = teach_df['teach_azimuth_timestamps']
# 4. (932,1) vertex timestamps
teach_vertex_timestamps = teach_df['teach_vertex_timestamps']
# 5. Pose at each vertex: (932,4,4)
teach_vertex_transforms = teach_df['teach_vertex_transforms']
# 6. teach vertext time
teach_times = teach_df['teach_times']


# load the repeat data
repeat_df = np.load(os.path.join(REPEAT_FOLDER, f"repeat.npz"),allow_pickle=True)
# in the repeat
repeat_times = repeat_df['repeat_times']
repeat_polar_imgs = repeat_df['repeat_polar_imgs']
repeat_azimuth_angles = repeat_df['repeat_azimuth_angles']
repeat_azimuth_timestamps = repeat_df['repeat_azimuth_timestamps']
repeat_vertex_timestamps = repeat_df['repeat_vertex_timestamps']
repeat_edge_transforms = repeat_df['repeat_edge_transforms']
vtr_estimated_ptr = repeat_df['dist']


# load the result data
result_df = np.load(os.path.join(RESULT_FOLDER, f"result.npz"),allow_pickle=True)
vtr_norm = result_df['vtr_norm']
gps_norm = result_df['gps_norm']
dir_norm = result_df['dir_norm']
direct_se2_pose = result_df['direct_se2_pose']
vtr_se2_pose = result_df['vtr_se2_pose']
gps_teach_pose = result_df['gps_teach_pose']
gps_repeat_pose = result_df['gps_repeat_pose']

timestamp_association= result_df['timestamp_association']

vtr_pte = result_df['vtr_pte']
vtr_loc_error = result_df['vtr_loc_error']
direct_pte = result_df['direct_pte']
direct_loc_error = result_df['direct_loc_error']


# okay now we want to output areas where loc errors differences are high and give the timestamps
# we will use the vtr_loc_error and direct_loc_error to find the high error regions
high_error_threshold = 0.05  # Define a threshold for high localization error
high_error_indices = np.where(np.abs(vtr_loc_error - direct_loc_error) > high_error_threshold)[0]

print(f"High error indices: {high_error_indices}")

# I want to group these indices into regions maybe have some lee way where it does not have to be consecutive but within a certain range of indeices 
window = 30
regions = []
current_region = []

for i in range(len(high_error_indices)):
    if not current_region:
        current_region.append(high_error_indices[i])
    else:
        # Check if the current index is within the window of the last index in the current region
        if high_error_indices[i] - current_region[-1] <= window:
            current_region.append(high_error_indices[i])
        else:
            # If not, save the current region and start a new one
            regions.append(current_region)
            current_region = [high_error_indices[i]]

regions.append(current_region)  # Add the last region if it exists

# how many regions do we have?
print(f"Number of regions with high localization error differences: {len(regions)}")
print(f"Regions with high localization error differences: {regions}")

# I want to print the starting and ending timestamps of these regions using repeat_times
for region in regions:
    start_index = region[0]
    end_index = region[-1]
    start_time = repeat_times[start_index][0]
    end_time = repeat_times[end_index][0]
    print(f"Region from index {start_index} to {end_index}: Start time: {start_time}, End time: {end_time}")
