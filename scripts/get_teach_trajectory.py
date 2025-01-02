import os
import matplotlib.pyplot as plt
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse

import yaml

import sys

# print current working directory
print("Current working dir", os.getcwd())

# Insert path at index 0 so it's searched first
sys.path.insert(0, "scripts")

from radar.utils.helper import get_xyt_gps

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

def wrap_angle(theta):
    if(theta > np.pi):
        return theta - 2*np.pi
    elif(theta < -np.pi):
        return theta + 2*np.pi
    else:
        return theta


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

    offline_graph_dir = pose_graph_path
    factory = Rosbag2GraphFactory(offline_graph_dir)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

    g_utils.set_world_frame(test_graph, test_graph.root)

    v_start = test_graph.root

    v_start = test_graph.get_vertex((0, 0))

    path_matrix = vtr_path.path_to_matrix(test_graph, PriviledgedIterator(v_start))
    print(path_matrix.shape)

    x = []
    y = []
    t = []

    traj = np.empty((0, 4))

    for v, e in PriviledgedIterator(v_start):
        # seems like this is from world to vertex
        # T_v_w = v.T_v_w

        C_v_w = v.T_v_w.C_ba()
        # print(v.T_v_w.matrix())
        # print("The rotation matrix is: ", C_v_w)
        x_k = v.T_v_w.r_ba_ina()[0]
        y_k = v.T_v_w.r_ba_ina()[1]
        t_k = v.stamp / 1e9 
        roll,pitch,yaw = rotation_matrix_to_euler_angles(C_v_w)

        yaw = wrap_angle(yaw)

        x.append(x_k[0])
        y.append(y_k[0])
        t.append(t_k)

        to_append = np.array([t_k,x_k[0], y_k[0], yaw])

        # print(to_append)
        traj = np.vstack((traj, to_append))
    
    print("for loop done the shape of traj is ",traj.shape)

    # now we want to process the gt
    x_gt,y_gt,t_gt = get_xyt_gps(teach_rosbag_path)
    traj_gt = np.dstack((t_gt,x_gt,y_gt))[0]

    print("The shape of the ground truth is: ", traj_gt.shape)
    
    if PLOT:
        plt.figure(0)
        # print("The length of x is ", len(x))
        # plt.plot(x, y, label="Teach", linewidth=5)
        plt.plot(x_gt, y_gt, label="Ground Truth", linewidth=5)
        plt.axis('equal')

        plt.show()

    if SAVE:
        traj_folder = os.path.join(save_folder, "traj")
        if not os.path.exists(traj_folder):
            os.makedirs(traj_folder)
            print(f"Folder '{traj_folder}' created.")

        np.savetxt(os.path.join(traj_folder, "teach_traj_estimated.txt"), traj, delimiter=",")
        np.savetxt(os.path.join(traj_folder, "teach_traj_gt.txt"), traj_gt, delimiter=",")
        print("Teach trajectory saved.")

    print("Done.")
