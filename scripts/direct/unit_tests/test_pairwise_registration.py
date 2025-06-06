# test_pairwise registration.py
# essentially this is one to one image registration 
# this script will take in two image path and then do pairwise or localmap registration 

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
print("Current working dir", os.getcwd())

from deps.path_tracking_error.fcns import *
# from scripts.radar.utils.helper import *

# # point cloud vis
# from sensor_msgs_py.point_cloud2 import read_points
# # import open3d as o3d
from pylgmath import Transformation
# from vtr_utils.plot_utils import *
# import time
import yaml
import scripts.direct.gp_doppler as gpd
import cv2
import torch
import torchvision

import pyboreas as pb

# from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
# from scripts.utils import *



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

config = load_config(os.path.join(parent_folder,'scripts/direct/direct_configs/direct_config_sam.yaml'))


db_bool = config['bool']
SAVE = db_bool.get('SAVE')
PLOT = db_bool.get('PLOT')
DEBUG = db_bool.get('DEBUG')
DEBUG = False
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

# load data 
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


# ---------- code start here -----------

# start the gp estimator
gp_state_estimator_1 = gpd.GPStateEstimator(config_warthog, radar_resolution)
gp_state_estimator_2 = gpd.GPStateEstimator(config_warthog, radar_resolution)


image1_path = '/home/samqiao/ASRL/vtr3_testing/scripts/direct/unit_tests/unit_test_data/1738179490.9871147_scan.png'
image2_path = '/home/samqiao/ASRL/vtr3_testing/scripts/direct/unit_tests/unit_test_data/1738179491.236846.png'
# Load the images
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

# show images and its shape
print("Image 1 shape:", img1.shape)
print("Image 2 shape:", img2.shape)

# Convert images to grayscale
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # Show images
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(img1_gray, cmap='gray')
# plt.title('Cart Image 1')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(img2_gray, cmap='gray')
# plt.title('Cart Image 2')
# plt.axis('off')

# plt.show()

target_timestamp = 1738179490.9871147
# lets find the polar information from the cartesian images
idx =  np.argmin(np.abs(teach_times - target_timestamp))
print("Index of closest timestamp:", idx)
idx = 1
print("The teach vertex timestamp:", teach_vertex_timestamps[1][0])

# Get the polar image and azimuth angles for the closest timestamp
polar_image = teach_polar_imgs[idx]
azimuth_angles = teach_azimuth_angles[idx]
azimuth_timestamps = teach_azimuth_timestamps[idx]

# Convert the polar image to cartesian
cart_image = radar_polar_to_cartesian(polar_image, azimuth_angles, radar_resolution, cart_resolution=cart_resolution)

# # Show the cartesian image
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(cart_image, cmap='gray')
# plt.title('Cartesian Image from Polar Data')
# plt.axis('off')
# plt.show()

# now we will do teach image to teach local map registration
# we will constrast the optimization score with teach image to teach image registration
class RadarFrame:
    def __init__(self, polar, azimuths, timestamps):
        self.polar = polar[:, :].astype(np.float32) / 255.0
        self.azimuths=azimuths
        self.timestamps=timestamps.flatten().astype(np.int64) 

teach_frame = RadarFrame(polar_image, azimuth_angles, azimuth_timestamps)

def preprocess_frame(frame):
    """
    Preprocess the radar frame by converting polar data to cartesian.
    """
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        timestamps = torch.tensor(frame.timestamps.flatten()).to(device).squeeze()
        # Prepare the data in torch
        azimuths = torch.tensor(frame.azimuths.flatten()).to(device).float()
        nb_azimuths = torch.tensor(len(frame.azimuths.flatten())).to(device) # number of azimuths 400
        # motion_model.setTime(self.timestamps, self.timestamps[0])
        
        # # Initialise the direction vectors
        # # only use for doppler
        # dirs = torch.empty((self.nb_azimuths, 2), device=self.device) # 400 by 2
        # dirs[:, 0] = torch.cos(self.azimuths)
        # dirs[:, 1] = torch.sin(self.azimuths)
        # self.vel_to_bin_vec = self.vel_to_bin*dirs
        # # doppler possible

        polar_intensity = torch.tensor(frame.polar).to(device)
        # print("polar_intensity shape", self.polar_intensity.shape)

        # normalization and smoothing
        polar_std = torch.std(polar_intensity, dim=1)
        polar_mean = torch.mean(polar_intensity, dim=1)
        polar_intensity -= (polar_mean.unsqueeze(1) + 2*polar_std.unsqueeze(1))
        polar_intensity[polar_intensity < 0] = 0
        polar_intensity = torchvision.transforms.functional.gaussian_blur(polar_intensity.unsqueeze(0), (9,1), 3).squeeze()
        polar_intensity /= torch.max(polar_intensity, dim=1, keepdim=True)[0]
        polar_intensity[torch.isnan(polar_intensity)] = 0

        frame_cart = pb.utils.radar.radar_polar_to_cartesian(frame.azimuths.astype(np.float32), polar_intensity.detach().cpu().numpy(), 0.040308, 0.2384, 640, False, True)

        # visualize the cartesian image
        plt.imshow(frame_cart, cmap='gray')
        plt.title('Cartesian Image from Polar Data')
        plt.axis('off')
        plt.show()

        return polar_intensity,frame_cart, azimuths, timestamps, nb_azimuths


# cart_frame = preprocess_frame(teach_frame)


# # first we will do teach image to teach local map registration
teach_local_map_file = image2_path
teach_local_map = load_local_map(teach_local_map_file)
state = gp_state_estimator_1.toLocalMapRegistration(teach_local_map, teach_frame, teach_frame)

print("State after teach image to teach local map registration:", state)
# now we will do teach image to teach image registration
print("-----------------------perform teach image to teach image registration-----------------------")
state = gp_state_estimator_2.pairwiseRegistration(teach_frame, teach_frame)
print("State after teach image to teach image registration:", state)




