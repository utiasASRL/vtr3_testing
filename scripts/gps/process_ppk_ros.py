import rosbags
# import cv2
# from cv_bridge import CvBridge
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from matplotlib import pyplot as plt

import utm

# print current working directory
import os
print("Current working dir", os.getcwd())

import sys
# Insert path at index 0 so it's searched first
parent_folder = "/home/samqiao/ASRL/vtr3_testing"
sys.path.insert(0, parent_folder)
# sys.path.insert(0, "scripts")
# sys.path.insert(0, "deps")

# from radar.utils.helper import get_xyt_gps
from deps.path_tracking_error.fcns import *

from scipy import interpolate
import csv
import yaml

print("Current working dir", os.getcwd())

def plot_3D(x,y,z):
    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with color representing time progression
    scatter = ax.scatter(x, y, z, c=np.arange(len(x)), cmap='viridis', s=10, label='Trajectory Points')
    ax.plot(x, y, z, color='gray', alpha=0.4, linewidth=0.8, label='Path')

    # Set Equal Aspect Ratio
    def set_axes_equal(ax):
        """Set equal aspect ratio for 3D plots."""
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        z_limits = ax.get_zlim()

        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]
        max_range = max(x_range, y_range, z_range) / 2.0

        mid_x = np.mean(x_limits)
        mid_y = np.mean(y_limits)
        mid_z = np.mean(z_limits)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Apply equal axis scaling
    set_axes_equal(ax)


    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend()

    # Add a color bar to show the progression over time
    plt.colorbar(scatter, label='Time Step', pad=0.1)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.title('3D Trajectory Visualization')
    plt.show()

def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file.

    :param config_path: Path to the YAML configuration file.
    :return: A dictionary representing the configuration.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


config = load_config(os.path.join(parent_folder,'scripts/direct/direct_config.yaml'))

# Access database configuration
db = config['radar_data']['parking']
db_rosbag_path = db.get('rosbag_path')

# teach_rosbag_path = db_rosbag_path.get('teach')

global repeat
repeat = 1
# repeat_rosbag_path = db_rosbag_path.get(f'repeat{repeat}') # dont think this is needed

# # for pose graph
# pose_graph_path = db.get('pose_graph_path').get('paring_t3_r4')
# print("pose graph path:",pose_graph_path)

db_bool = config['bool']
SAVE = db_bool.get('SAVE')
SAVE = False
print("SAVE:",SAVE)
PLOT = db_bool.get('PLOT')
DEBUG = db_bool.get('DEBUG')
print("PLOT:",PLOT)
print("DEBUG:",DEBUG)

result_folder = config.get('output')

# change here
out_path_folder = os.path.join(result_folder,f"parking_t3_r4") # change path here
if not os.path.exists(out_path_folder):
    os.makedirs(out_path_folder)
    print(f"Folder '{out_path_folder}' created.")
else:
    print(f"Folder '{out_path_folder}' already exists.")    

# basically, an array of (n by 3) ros_time, x, y
teach_ppk_folder = config['ppk']['parking']['teach']
repeat_ppk_folder = config['ppk']['parking']['repeat']
print("teach_ppk_folder:",teach_ppk_folder)
print("repeat_ppk_folder:",repeat_ppk_folder)


teach_ppk = os.path.join(teach_ppk_folder, "parking_t3.txt")
teach_ppk_ros = os.path.join(teach_ppk_folder, "parking_t3_ros.txt")
repeat_ppk = os.path.join(repeat_ppk_folder, "parking_t4.txt")
repeat_ppk_ros = os.path.join(repeat_ppk_folder, "parking_t4_ros.txt")

# lets get teach first
t_teach_ppk = read_gps_ros_txt(teach_ppk_ros)
print("t_teach_ppk shape:",t_teach_ppk.shape)
x_teach_ppk, y_teach_ppk = read_PPK_file_sam(teach_ppk)
print("x_teach_ppk shape:",x_teach_ppk.shape)

r2_pose_teach_ppk = np.hstack((t_teach_ppk.reshape(-1,1), x_teach_ppk.reshape(-1,1), y_teach_ppk.reshape(-1,1),np.zeros_like(x_teach_ppk.reshape(-1,1))))
print("r2_pose_teach_ppk shape:",r2_pose_teach_ppk.shape)

teach_ppk_length_3d = get_path_distance_from_gps(x_teach_ppk, y_teach_ppk)
print("teach_ppk_length_3d:",teach_ppk_length_3d)

# lets get repeat
t_repeat_ppk = read_gps_ros_txt(repeat_ppk_ros)
print("t_repeat_ppk shape:",t_repeat_ppk.shape)
x_repeat_ppk, y_repeat_ppk = read_PPK_file_sam(repeat_ppk)
print("x_repeat_ppk shape:",x_repeat_ppk.shape)
r2_pose_repeat_ppk = np.hstack((t_repeat_ppk.reshape(-1,1), x_repeat_ppk.reshape(-1,1), y_repeat_ppk.reshape(-1,1),np.zeros_like(x_repeat_ppk.reshape(-1,1))))
print("r2_pose_repeat_ppk shape:",r2_pose_repeat_ppk.shape)
repeat_ppk_length_3d = get_path_distance_from_gps(x_repeat_ppk, y_repeat_ppk)
print("repeat_ppk_length_3d:",repeat_ppk_length_3d)


plt.figure(0)
plt.plot(x_teach_ppk, y_teach_ppk,  label='PPK GPS Teach')
plt.plot(x_repeat_ppk, y_repeat_ppk,  label='PPK GPS Repeat')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.axis('equal')
plt.title('PPK GPS Teach')
plt.legend()
plt.grid()
plt.show()




# plot_3D(x_teach, y_teach, z_teach)
teach_save_folder = os.path.join(out_path_folder, "teach")
if not os.path.exists(teach_save_folder):
    os.makedirs(teach_save_folder)
    print(f"Folder '{teach_save_folder}' created.")
np.savez(os.path.join(teach_save_folder, "teach_ppk.npz"), r2_pose_teach_ppk=r2_pose_teach_ppk)


repeat_save_folder = os.path.join(out_path_folder, "repeat")
if not os.path.exists(repeat_save_folder):
    os.makedirs(repeat_save_folder)
    print(f"Folder '{repeat_save_folder}' created.")
np.savez(os.path.join(repeat_save_folder, "repeat_ppk.npz"), r2_pose_repeat_ppk=r2_pose_repeat_ppk)

print("saved!")