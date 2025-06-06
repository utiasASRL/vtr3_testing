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
# from scripts.radar.utils.helper import *

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

from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images

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


def radar_polar_to_cartesian(fft_data, azimuths, radar_resolution, cart_resolution=0.2384, cart_pixel_width=640,
                             interpolate_crossover=False, fix_wobble=True):
    # TAKEN FROM PYBOREAS
    """Convert a polar radar scan to cartesian.
    Args:
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        fft_data (np.ndarray): Polar radar power readings
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        cart_resolution (float): Cartesian resolution (metres per pixel)
        cart_pixel_width (int): Width and height of the returned square cartesian output (pixels)
        interpolate_crossover (bool, optional): If true interpolates between the end and start  azimuth of the scan. In
            practice a scan before / after should be used but this prevents nan regions in the return cartesian form.

    Returns:
        np.ndarray: Cartesian radar power readings
    """
    # print("in radar_polar_to_cartesian")
    # Compute the range (m) captured by pixels in cartesian scan
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    
    # Compute the value of each cartesian pixel, centered at 0
    coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)

    Y, X = np.meshgrid(coords, -1 * coords)
    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

    # Interpolate Radar Data Coordinates
    azimuth_step = (azimuths[-1] - azimuths[0]) / (azimuths.shape[0] - 1)
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution

    # print("------")
    # print("sample_angle.shape",sample_angle.shape)
    # print("azimuths[0]",azimuths[0])
    # print("azimuth step shape" ,azimuth_step.shape)

    sample_v = (sample_angle - azimuths[0]) / azimuth_step
    # This fixes the wobble in the old CIR204 data from Boreas
    M = azimuths.shape[0]
    azms = azimuths.squeeze()
    if fix_wobble:
        c3 = np.searchsorted(azms, sample_angle.squeeze())
        c3[c3 == M] -= 1
        c2 = c3 - 1
        c2[c2 < 0] += 1
        a3 = azms[c3]
        diff = sample_angle.squeeze() - a3
        a2 = azms[c2]
        delta = diff * (diff < 0) * (c3 > 0) / (a3 - a2 + 1e-14)
        sample_v = (c3 + delta).astype(np.float32)

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    return cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR)

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
SET_INITIAL_GUESS = db_bool.get('SET_INITIAL_GUESS')

print("Bool values from config:")
print("SAVE:", SAVE)
print("PLOT:", PLOT)
print("DEBUG:", DEBUG)
print("USE_LOCAL_MAP:", USE_LOCAL_MAP)
print("SET_INITIAL_GUESS:", SET_INITIAL_GUESS)

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


# load vtr posegraph results
TEACH_FOLDER = os.path.join(out_path_folder, "teach")
REPEAT_FOLDER = os.path.join(out_path_folder, f"repeat")

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

if DEBUG:
    print("teach_polar_imgs shape:", teach_polar_imgs.shape)
    print("teach_azimuth_angles shape:", teach_azimuth_angles.shape)
    print("teach_azimuth_timestamps shape:", teach_azimuth_timestamps.shape)
    print("teach_vertex_timestamps shape:", teach_vertex_timestamps.shape)
    # print teach duration
    print("teach duration:", teach_times[-1] - teach_times[0])
    print("teach_vertex_transforms shape:", teach_vertex_transforms.shape)

repeat_df = np.load(os.path.join(REPEAT_FOLDER, f"repeat.npz"),allow_pickle=True)
# in the repeat
repeat_times = repeat_df['repeat_times']
repeat_polar_imgs = repeat_df['repeat_polar_imgs']
repeat_azimuth_angles = repeat_df['repeat_azimuth_angles']
repeat_azimuth_timestamps = repeat_df['repeat_azimuth_timestamps']
repeat_vertex_timestamps = repeat_df['repeat_vertex_timestamps']
repeat_edge_transforms = repeat_df['repeat_edge_transforms']

vtr_estimated_ptr = repeat_df['dist']

if DEBUG:
    print("repeat_polar_imgs shape:", repeat_polar_imgs.shape)
    print("repeat_azimuth_angles shape:", repeat_azimuth_angles.shape)
    print("repeat_azimuth_timestamps shape:", repeat_azimuth_timestamps.shape)
    print("repeat_vertex_timestamps shape:", repeat_vertex_timestamps.shape)
    # print repeat duration
    print("repeat duration:", repeat_times[-1] - repeat_times[0])
    print("repeat_edge_transforms shape:", repeat_edge_transforms.shape)
    print("vtr_estimated_ptr shape:", vtr_estimated_ptr.shape)


# now we will get a baseline quick norm 
# also I need to verify the xyz in a plane
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


# the things that I need
# 1. repeat_edge_transforms = repeat_df['repeat_edge_transforms']
print("repeat times shape:", repeat_times.shape)
print("repeat_edge_transforms shape:", repeat_edge_transforms.shape)

gps_norm = []
vtr_norm = []

vtr_x_error = []
vtr_y_error = []
vtr_yaw_error = []

# maybe lets run it for 10 times
print("begin loop")
for repeat_vertex_idx in range(0,repeat_times.shape[0]):
    print("------------------ repeat idx: ", repeat_vertex_idx,"------------------")
    teach_vertex_time = teach_times[repeat_vertex_idx]
    repeat_vertex_time = repeat_times[repeat_vertex_idx]

    print("teach vertex time:", teach_vertex_time[0])
    print("repeat vertex time:", repeat_vertex_time[0])

    # populate vtr norm with the 2d euclidean distance
    print("repeat_edge_transforms shape:", repeat_edge_transforms.shape)

    T_teach_repeat_edge = repeat_edge_transforms[repeat_vertex_idx][0][repeat_vertex_time[0]]

    # print("T_teach_repeat_edge shape:", T_teach_repeat_edge.matrix())
    
    r_repeat_teach_in_teach = T_teach_repeat_edge.inverse().r_ba_ina()
    print("r_repeat_teach_in_teach shape:", r_repeat_teach_in_teach.shape)

    vtr_norm.append(np.linalg.norm(r_repeat_teach_in_teach[0:2]))
    print("vtr norm:", vtr_norm[repeat_vertex_idx])

    # do the same for gps
    def get_closest_gps_measurement(r2_pose_ppk,t_query):
        """
        Find the closest GPS measurement to a given timestamp.

        :param r2_pose_ppk: Array of GPS measurements.
        :param t_query: Timestamp to find the closest measurement for.
        :return: Closest GPS measurement and its index.
        """
        idx = np.argmin(np.abs(r2_pose_ppk[:, 0] - t_query))
        return r2_pose_ppk[idx], idx
    
    # get the closest gps measurement
    teach_ppk, idx = get_closest_gps_measurement(r2_pose_teach_ppk_dirty, teach_vertex_time)
    print("teach_ppk idx", idx)
    repeat_ppk,idx= get_closest_gps_measurement(r2_pose_repeat_ppk_dirty, repeat_vertex_time)
    print("repeat_ppk idx", idx)
    print("teach_ppk:", teach_ppk)
    print("repeat_ppk:", repeat_ppk)

    # calculate the norm
    gps_norm.append(np.linalg.norm(teach_ppk[1:3] - repeat_ppk[1:3]))
    print("gps norm:", gps_norm[repeat_vertex_idx])
    # print("T_teach_repeat_edge_options:", T_teach_repeat_edge_options)

class RadarFrame:
    def __init__(self, polar, azimuths, timestamps):
        self.polar = polar[:, :].astype(np.float32) / 255.0
        self.azimuths=azimuths
        self.timestamps=timestamps.flatten().astype(np.int64) 

dir_norm = []
vtr_se2_pose = []
direct_se2_pose = []

# load all the local maps of the teach path
# open the directory
teach_local_maps_path = config["radar_data"]["grassy"]["local_maps_path"]
print(teach_local_maps_path)
teach_local_maps_files = os.listdir(teach_local_maps_path)

teach_local_maps = {}
for file in teach_local_maps_files:
    if file.endswith(".png"):
        file_path = os.path.join(teach_local_maps_path, file)
        teach_local_maps[file.replace(".png","")] = file_path

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

# print("------ dir norm ------")
# now we can set up the direct localization stuff
for repeat_vertex_idx in range(0,repeat_times.shape[0]):
    print("------------------ repeat idx: ", repeat_vertex_idx,"------------------")
    repeat_vertex_time = repeat_times[repeat_vertex_idx]

    teach_cv_scan_polar = teach_polar_imgs[repeat_vertex_idx]
    teach_scan_azimuth_angles = teach_azimuth_angles[repeat_vertex_idx][:] # need to add : so the first element is correct
    teach_scan_timestamps = teach_azimuth_timestamps[repeat_vertex_idx]

    print("teach azimuth angles shape:", teach_scan_azimuth_angles.shape)
    print("the first teach azimuth angle:", teach_scan_azimuth_angles[0])

    teach_vertex_time = teach_vertex_timestamps[repeat_vertex_idx]
    print("sam: teach vertex time:", teach_vertex_time[0])

    teach_cv_scan_cartesian = radar_polar_to_cartesian(teach_cv_scan_polar,teach_scan_azimuth_angles, radar_resolution, cart_resolution, 640)
    

    repeat_cv_scan_polar = repeat_polar_imgs[repeat_vertex_idx]
    repeat_scan_azimuth_angles = repeat_azimuth_angles[repeat_vertex_idx][:]
    repeat_scan_timestamps = repeat_azimuth_timestamps[repeat_vertex_idx]

    repeat_cv_scan_cartesian = radar_polar_to_cartesian(repeat_cv_scan_polar,repeat_scan_azimuth_angles, radar_resolution, cart_resolution, 640)

    teach_frame = RadarFrame(teach_cv_scan_polar, teach_scan_azimuth_angles, teach_scan_timestamps.reshape(-1,1))
    repeat_frame = RadarFrame(repeat_cv_scan_polar, repeat_scan_azimuth_angles, repeat_scan_timestamps.reshape(-1,1))

    # we can use teach and repeat result as a intial guess
    T_teach_repeat_edge = repeat_edge_transforms[repeat_vertex_idx][0][repeat_vertex_time[0]]    
    T_teach_repeat_edge_in_radar = T_radar_robot @ T_teach_repeat_edge @ T_radar_robot.inverse()
    r_repeat_teach_in_teach = T_teach_repeat_edge_in_radar.inverse().r_ba_ina() # inverse?

    # we might need to transform r_repeat_teach to the radar frame
    roll, pitch, yaw = rotation_matrix_to_euler_angles(T_teach_repeat_edge.C_ba().T) 

    # state = gp_state_estimator.pairwiseRegistration(teach_frame, repeat_frame)to_euler_angles(T_teach_repeat_edge.C_ba().T) # ba means teach_repeat
    r_repeat_teach_in_teach[2] = wrap_angle(yaw)
   

    vtr_se2_pose.append(r_repeat_teach_in_teach.T[0])
    intial_guess = torch.from_numpy(np.squeeze(r_repeat_teach_in_teach)).to('cuda')
    # print("intial_guess shape:", intial_guess.shape)

    if SET_INITIAL_GUESS:
        # set the state to the intial guess
        gp_state_estimator.setIntialState(intial_guess)

    # teach_timestamp = teach_times[repeat_vertex_idx]

    if USE_LOCAL_MAP:
        teach_local_map_file, _ = find_closest_local_map(teach_local_maps, teach_times[repeat_vertex_idx][0])
        # print("teach_local_map_file:", teach_local_map_file)

        teach_local_map = load_local_map(teach_local_map_file)

        
        state = gp_state_estimator.toLocalMapRegistration(teach_local_map, teach_frame, teach_frame)

    else:
        state = gp_state_estimator.pairwiseRegistration(teach_frame, teach_frame)


    direct_se2_pose.append(state)
    norm_state = np.linalg.norm(state[0:2])

    dir_norm.append(norm_state)

    print("r_repeat_teach_in_teach:", r_repeat_teach_in_teach.T[0])
    print("direct estimated state:", state)

    # if repeat_vertex_idx == 2:
    #     break



result_folder = os.path.join(out_path_folder, "direct")
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
    print(f"Folder '{result_folder}' created.")


# save the results
vtr_norm = np.array(vtr_norm)
gps_norm = np.array(gps_norm)
dir_norm = np.array(dir_norm)
direct_se2_pose = np.array(direct_se2_pose)
vtr_se2_pose = np.array(vtr_se2_pose)

# also want to save the gps here t,x,y
gps_teach_pose = r2_pose_teach_ppk_dirty
gps_repeat_pose = r2_pose_repeat_ppk_dirty

# need to get gps_path_tracking_error
# step 1: make a path matrix
# step 2: accumulate the signed distance

print("vtr_norm shape:", vtr_norm.shape)
print("gps_norm shape:", gps_norm.shape)
print("dir_norm shape:", dir_norm.shape)
print("gps_teach_pose shape:", gps_teach_pose.shape)
print("gps_repeat_pose shape:", gps_repeat_pose.shape)
print("direct_se2_pose shape:", direct_se2_pose.shape)
print("vtr_se2_pose shape:", vtr_se2_pose.shape)


np.savez(os.path.join(result_folder, "result.npz"),
         vtr_norm=vtr_norm,
         gps_norm=gps_norm,
         dir_norm=dir_norm,
         direct_se2_pose=direct_se2_pose,
         vtr_se2_pose=vtr_se2_pose, 
         gps_teach_pose=gps_teach_pose,
         gps_repeat_pose=gps_repeat_pose)
    