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

# from deps.path_tracking_error.fcns import *
from scripts.radar.utils.helper import *

# # point cloud vis
# from sensor_msgs_py.point_cloud2 import read_points
# # import open3d as o3d
from pylgmath import Transformation
# from vtr_utils.plot_utils import *
# import time

import yaml
import gp_doppler as gpd
import torch
import torchvision

# from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images

from utils import *


print("Current working dir", os.getcwd())

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
PLOT = db_bool.get('PLOT')
DEBUG = db_bool.get('DEBUG')
USE_LOCAL_MAP = db_bool.get('USE_LOCAL_MAP')
LOCAL_TO_LOCAL = db_bool.get('LOCAL_TO_LOCAL')
UNDISTORTION = db_bool.get('UNDISTORTION')
SET_INITIAL_GUESS = db_bool.get('SET_INITIAL_GUESS')

print("Bool values from config:")
print("SAVE:", SAVE)
print("PLOT:", PLOT)
print("DEBUG:", DEBUG)
# print("USE_LOCAL_MAP:", USE_LOCAL_MAP)
# print("UNDISTORTION:", UNDISTORTION)
# print("SET_INITIAL_GUESS:", SET_INITIAL_GUESS)

result_folder = config.get('output')

# change here
out_path_folder = os.path.join(result_folder,f"grassy_t2_r3/")
if not os.path.exists(out_path_folder):
    os.makedirs(out_path_folder)
    print(f"Folder '{out_path_folder}' created.")
else:
    print(f"Folder '{out_path_folder}' already exists.")    


# radar parameters
radar_resolution = config['radar_resolution']
cart_resolution = config['cart_resolution']
min_range = config['min_r']
max_range = config['max_r']

# config_warthog for dp state estimator
config_warthog = load_config(os.path.join(parent_folder,'scripts/direct/warthog_config.yaml'))

# start the gp estimator
gp_state_estimator = gpd.GPStateEstimator(config_warthog, radar_resolution)


# in this file, we will set up the toy problem but use DRO for everything 
# the output of this file the timestamps of the repeat scan and map scan
# the state estimates -> make sure they are in the same frame 

class RadarFrame:
    def __init__(self, polar, azimuths, timestamps):
        self.polar = polar[:, :].astype(np.float32) / 255.0
        self.azimuths=azimuths
        self.timestamps=timestamps.flatten().astype(np.int64)

# these are for teach
# load all the local maps of the teach path
# open the directory 
teach_local_maps_path = config["radar_data"]["grassy"]["teach_local_maps_path"]
print(teach_local_maps_path)
teach_local_maps_files = os.listdir(teach_local_maps_path)

teach_local_maps = {}
for file in teach_local_maps_files:
    if file.endswith(".png"):
        file_path = os.path.join(teach_local_maps_path, file)
        teach_local_maps[file.replace(".png","")] = file_path

teach_undistorted_scans_path = config["radar_data"]["grassy"]["teach_undistorted_scans_path"]
print(teach_undistorted_scans_path)
teach_undistorted_scans_files = os.listdir(teach_undistorted_scans_path)

teach_undistorted_scans = {}
for file in teach_undistorted_scans_files:
    if file.endswith(".npz"):
        file_path = os.path.join(teach_undistorted_scans_path, file)
        teach_undistorted_scans[file.replace(".npz","")] = file_path

# do the same for the repeat scans
repeat_undistorted_scans_path = config["radar_data"]["grassy"]["repeat_undistorted_scans_path"]
print(repeat_undistorted_scans_path)
repeat_undistorted_scans_files = os.listdir(repeat_undistorted_scans_path)

repeat_undistorted_scans = {}
for file in repeat_undistorted_scans_files:
    if file.endswith(".npz"):
        file_path = os.path.join(repeat_undistorted_scans_path, file)
        repeat_undistorted_scans[file.replace(".npz","")] = file_path


# load the undistorted scans as well
def load_local_map(file_path):
    """
    Load a local map from a file.

    :param file_path: Path to the local map file.
    :return: Loaded local map.
    """
    # Assuming the local map is an image, you can use OpenCV or PIL to load it
    # For example, using OpenCV:
    local_map = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return local_map.astype(np.float32) / 255.0

def load_undistorted_scan(file_path):
    """
    Load an undistorted scan from a file.

    :param file_path: Path to the undistorted scan file.
    :return: Loaded undistorted scan.
    """
    # Assuming the undistorted scan is stored in a .npz file
    data = np.load(file_path)
    polar = data['polar_target']
    azimuths = data['azimuths']
    timestamps = data['timestamps']
    
    return RadarFrame(polar, azimuths, timestamps)

def find_closest_local_map(teach_local_maps, timestamp):
    """
    Find the closest local map to a given timestamp.

    :param teach_local_maps: Dictionary of local maps.
    :param timestamp: Timestamp to find the closest local map for.
    :return: Closest local map and its key.
    """
    print("looking for: ", timestamp)
    closest_key = min(teach_local_maps.keys(), key=lambda k: abs(float(k) - timestamp))
    print("closest key:", closest_key)
    return teach_local_maps[closest_key], closest_key

# here I wanto enforce a temporal constraint as well
def find_associated_map_timestamp_given_undistorted_scan(repeat_undistorted_scan, teach_local_maps, gt_teach, gt_repeat):
    """
    Find the associated map timestamp given an undistorted scan timestamp.

    :param teach_undistorted_scans: Dictionary of undistorted scans.
    :param timestamp: Timestamp of the undistorted scan.
    :return: Associated map timestamp.
    """
    print("looking for associated map for undistorted scan with timestamp:", timestamp)

    # we actually need to use the distance to find the association
    distorted_scan_timestamp = float(repeat_undistorted_scan.timestamps[-1])/1e6
    print("distorted scan timestamp:", distorted_scan_timestamp)

    # find the closst repeat gps timestamp
    closest_repeat_idx = np.argmin(np.abs(gt_repeat[:, 0] - distorted_scan_timestamp))
    
    print("closest repeat idx:", closest_repeat_idx)
    closest_repeat_timestamp = gt_repeat[closest_repeat_idx, 0]
    print("closest repeat timestamp:", closest_repeat_timestamp)

    repeat_gps_xyz = gt_repeat[closest_repeat_idx, 1:4]
    print("repeat gps xyz:", repeat_gps_xyz)

    # now we need to find the closest teach gps timestamp based on minimum distance with teach_gps
    distances = np.linalg.norm(gt_teach[:, 1:4] - repeat_gps_xyz, axis=1)
    closest_teach_idx = np.argmin(distances) 
    print("closest teach idx:", closest_teach_idx)
    print("the min distance is:", distances[closest_teach_idx],"meters")   
    closest_teach_timestamp = gt_teach[closest_teach_idx, 0]
    print("closest teach timestamp:", closest_teach_timestamp)

    # now based on the closest teach timestamp, we can find the local map timestamp and return it
    closest_local_map_key = min(teach_local_maps.keys(), key=lambda k: abs(float(k) - closest_teach_timestamp))
    print("closest local map key:", closest_local_map_key)
    return  closest_local_map_key

##### now here is the gps GT
# now we will get a baseline quick norm 
# also I need to verify the xyz in a plane
# load vtr posegraph results
TEACH_FOLDER = os.path.join(out_path_folder, "teach")
REPEAT_FOLDER = os.path.join(out_path_folder, f"repeat")

teach_ppk_df = np.load(os.path.join(TEACH_FOLDER, "teach_ppk.npz"),allow_pickle=True)
if not os.path.exists(os.path.join(TEACH_FOLDER, "teach_ppk.npz")):
    print("teach_ppk.npz not found")
    exit(1)
r2_pose_teach_ppk_dirty = teach_ppk_df['r2_pose_teach_ppk']

repeat_ppk_df = np.load(os.path.join(REPEAT_FOLDER, "repeat_ppk.npz"),allow_pickle=True)
if not os.path.exists(os.path.join(REPEAT_FOLDER, "repeat_ppk.npz")):
    print("repeat_ppk.npz not found")
    exit(1)
r2_pose_repeat_ppk_dirty = repeat_ppk_df['r2_pose_repeat_ppk']

print("teach_ppk shape:", r2_pose_teach_ppk_dirty.shape)
print("repeat_ppk shape:", r2_pose_repeat_ppk_dirty.shape)

# note to myself:  we probably need to interpolate the gps data to get the timestamps right but since we are drving slowly, we can just use the closest one


# print what I have in the teach_undistorted_scans
print(len(teach_undistorted_scans.keys()))
# loop through all the undistorted scans actually 

dro_se2_pose = []
repeat_scan_stamps = []

timestamp_association = []

nframe = 0

for timestamp in sorted(repeat_undistorted_scans.keys()):
    print(f"------------------------------Processing undistorted scan with timestamp: {timestamp} -----------------------------frame number: {nframe}")
    # load the undistorted scan
    undistorted_scan = load_undistorted_scan(repeat_undistorted_scans[timestamp])
    print("undistorted scan polar shape:", undistorted_scan.polar.shape)
    print("undistorted scan azimuths shape:", undistorted_scan.azimuths.shape)
    print("undistorted scan timestamps shape:", undistorted_scan.timestamps.shape)

    # find the associated map timestamp
    associated_map_timestamp = find_associated_map_timestamp_given_undistorted_scan(undistorted_scan, teach_local_maps, r2_pose_teach_ppk_dirty, r2_pose_repeat_ppk_dirty)

    # print("associated map timestamp:", associated_map_timestamp)

    # # we can see the cartesian image to verify
    import pyboreas as pb
    cart_target = pb.utils.radar.radar_polar_to_cartesian(undistorted_scan.azimuths, undistorted_scan.polar, 0.040308, 0.224, 640, False, True)

    # find the closest local map
    local_map_path, local_map_key = find_closest_local_map(teach_local_maps, float(associated_map_timestamp))

    timestamp_association.append({undistorted_scan.timestamps[-1]/1e6: local_map_key})  # save the association between undistorted scan timestamp and local map key

    # print("local map path:", local_map_path)
    local_map = load_local_map(local_map_path)
    # print("local map shape:", local_map.shape)
    
    # # # we can see the local map to verify lets actually plot the local map and the undistorted scan # how do I make it interactive?
    # plt.clf()  # 
  
    # plt.subplot(1, 2, 1)
    # plt.imshow(local_map, cmap='gray')
    # plt.title(f"Local Map {local_map_key}")
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(cart_target, cmap='gray')
    # plt.title(f"Undistorted Scan {timestamp}")
    # plt.axis('off')
    # plt.tight_layout()

    # fig.canvas.draw()

    # plt.pause(0.1)  # Pause to update the plot


    # fig.canvas.flush_events()
    # time.sleep(0.05)  # 
  


    # lets set up the gp state estimator
    state = gp_state_estimator.toLocalMapRegistration_dro(local_map,undistorted_scan)
    dro_se2_pose.append(state)
    repeat_scan_stamps.append(float(timestamp))
    
    print("State shape:", state.shape)

    print("direct estimated state:", state)

    nframe += 1

    # break
# now we need to find the association between the undistorted scan and the local map
# they are from different sequences

dro_se2_pose = np.array(dro_se2_pose)
repeat_scan_stamps = np.array(repeat_scan_stamps).reshape(-1, 1)
timestamp_association = np.array(timestamp_association).reshape(-1, 1)

print("Direct SE2 Pose shape:", dro_se2_pose.shape)
print("Repeat Scan Stamps shape:", repeat_scan_stamps.shape)

# now we plot the direct se2 pose in x,y, and yaw in 3 by 1 fashion

plt.figure(figsize=(10, 5))
plt.subplot(3, 1, 1)
plt.plot(repeat_scan_stamps, dro_se2_pose[:, 0], label='X Position', color='r')
plt.title('Direct SE2 Pose - X Position')
plt.xlabel('Timestamp')
plt.ylabel('X Position (m)')
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(repeat_scan_stamps, dro_se2_pose[:, 1], label='Y Position', color='g')
plt.title('Direct SE2 Pose - Y Position')
plt.xlabel('Timestamp')
plt.ylabel('Y Position (m)')
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(repeat_scan_stamps, dro_se2_pose[:, 2], label='Yaw', color='b')
plt.title('Direct SE2 Pose - Yaw')
plt.xlabel('Timestamp')
plt.ylabel('Yaw (rad)')
plt.grid()  

plt.tight_layout()

plt.show()


# save the direct se2 pose
# I would like to save a few things to avoid loading everything again
np.savez(os.path.join(out_path_folder, "direct/dro_toy.npz"), 
            dro_se2_pose=dro_se2_pose,
            repeat_scan_stamps=repeat_scan_stamps,
            timestamp_association=timestamp_association
            )
    
