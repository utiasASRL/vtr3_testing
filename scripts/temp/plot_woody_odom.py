import os
import matplotlib.pyplot as plt
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory

from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse

import sys

# Insert path at index 0 so it's searched first
sys.path.insert(0, "scripts")

from radar.utils.helper import *
# point cloud vis
from sensor_msgs_py.point_cloud2 import read_points
# import open3d as o3d
from pylgmath import Transformation
from vtr_utils.plot_utils import *
import time

import yaml
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images

from scripts.visualization.animation import *


print("Current working dir", os.getcwd())

def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix SO(3) to Euler angles (roll, pitch, yaw).

    Parameters:
        R (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
        tuple: A tuple containing roll, pitch, and yaw angles (in radians).
    """
    assert R.shape == (3, 3), "Input rotation matrix must be 3x3."

    # Check if the matrix is a valid rotation matrix
    if not np.allclose(np.dot(R.T, R), np.eye(3)) or not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError("Input matrix is not a valid rotation matrix.")

    # Extract the Euler angles
    pitch = -np.arcsin(R[2, 0])

    if np.isclose(np.cos(pitch), 0):
        # Gimbal lock case
        roll = 0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return roll, pitch, yaw

def wrap_to_pi(angle_rad):
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


# initlize the video writer
# Parameters for the video writer
frame_rate = 60.0  # Frames per second
frame_size = (512, 512)  # Frame size (width, height) of the video
codec = cv2.VideoWriter_fourcc(*'XVID')  # Video codec (XVID, MJPG, etc.)

parent_folder = "/home/samqiao/ASRL/vtr3_testing"

T_novatel_robot =  Transformation(T_ba = np.array([[1.000, 0.000, 0.000, 0.550],
  [0.000, 1.000 , 0.000, 0.000],
  [0.000 ,0.000, 1.000 , -1.057],
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


config = load_config(os.path.join(parent_folder,'scripts/direct/direct_config_sam.yaml'))

# # Access database configuration
# db = config['radar_data']['grassy']
# db_rosbag_path = db.get('rosbag_path')

# teach_rosbag_path = db_rosbag_path.get('teach')

# global repeat
# repeat = 1
# repeat_rosbag_path = db_rosbag_path.get(f'repeat{repeat}') # dont think this is needed

# for pose graph
pose_graph_path = "/home/samqiao/ASRL/vtr3/temp/paper/radar/0910_woody1/graph"

print("-------- begin pose graph processing --------")
# Pose graph processing
factory = Rosbag2GraphFactory(pose_graph_path)

test_graph = factory.buildGraph()
# plot_graph(test_graph)
# plt.show()
print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

g_utils.set_world_frame(test_graph, test_graph.root)

# I can certainly create a path matrix for the teach branch

v_start = test_graph.root

x_teach = []
y_teach = []
z_teach = []
t_teach = []

for v, e in PriviledgedIterator(v_start):
    vertex = v
    # x_teach.append(v.T_v_w.r_ba_ina()[0])
    # y_teach.append(v.T_v_w.r_ba_ina()[1])
    # z_teach.append(v.T_v_w.r_ba_ina()[2])
    t_teach.append(v.stamp / 1e9)

    T_gps_w = T_novatel_robot @ vertex.T_v_w
    r_gps = (T_gps_w).r_ba_ina()
    x_teach_gps = r_gps[0]
    y_teach_gps = r_gps[1]
    z_teach_gps = r_gps[2]

    x_teach.append(x_teach_gps)
    y_teach.append(y_teach_gps)
    z_teach.append(z_teach_gps)
    # C_v_w = v.T_v_w.C_ba()

    # roll, pitch, yaw = rotation_matrix_to_euler_angles(C_v_w)

    # # need to wrap the yaw angle
    # yaw = wrap_to_pi(yaw)

    # roll_teach.append(roll)
    # pitch_teach.append(pitch)
    # yaw_teach.append(yaw)


x_teach = np.array(x_teach)
y_teach = np.array(y_teach)
z_teach = np.array(z_teach)
t_teach = np.array(t_teach)

pose_woody_teach = np.squeeze(np.array([x_teach, y_teach, z_teach]).T)

print("pose_woody_teach shape:", pose_woody_teach.shape)
print(int(pose_woody_teach.shape[0]))

animator = Animator()
animator.set_woody_temp(pose_woody_teach)

animator.animation()

# animator.save(os.path.join(os.getcwd(), 'woody_odom.mp4'))

