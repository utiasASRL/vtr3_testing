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

from radar.utils.helper import *
from deps.path_tracking_error.fcns import *

# print current working directory
print("Current working dir", os.getcwd())

# Insert path at index 0 so it's searched first
sys.path.insert(0, "scripts")

from radar.utils.helper import get_xyt_gps

from plot_3d_traj import plot_3d_traj # for plotting 3d trajectory upon completion

parent_folder = "/home/samqiao/ASRL/vtr3_testing"

from scipy.interpolate import interp1d

def resample_trajectory(traj, target_length):
    # Create interpolation functions for x and y
    original_indices = np.linspace(0, 1, len(traj))
    f_x = interp1d(original_indices, traj[:, 0], kind='linear')
    f_y = interp1d(original_indices, traj[:, 1], kind='linear')
    
    # Generate new indices for resampling
    new_indices = np.linspace(0, 1, target_length)
    resampled_traj = np.column_stack((f_x(new_indices), f_y(new_indices)))
    return resampled_traj


def align_trajectories(odom, gt):
  # Compute the centroids of the trajectories
  centroid_odom = np.mean(odom, axis=0)
  centroid_gt = np.mean(gt, axis=0)

  # Center the trajectories
  odom_centered = odom - centroid_odom
  gt_centered = gt - centroid_gt

  # Compute the covariance matrix
  H = np.dot(gt_centered.T, odom_centered)

  # Compute the Singular Value Decomposition (SVD)
  U, S, Vt = np.linalg.svd(H)

  # Compute the rotation matrix
  R = np.dot(U, Vt)

  # # Ensure a proper rotation matrix (det(R) should be 1)
  # if np.linalg.det(R) < 0:
  #   Vt[2, :] *= -1
  #   R = np.dot(U, Vt)

  # Compute the translation
  t = centroid_odom - np.dot(centroid_gt, R)

  # print("R: ", R)
  # print("t: ", t)

  # Apply the rotation to traj1
  # Apply the translation
  gt_aligned = np.dot(gt, R)
  gt_aligned += t

  return gt_aligned


# def align_trajectories(est_traj, gt_traj):
#     """
#     Aligns `est_traj` (estimated trajectory) to `gt_traj` (ground truth) using SE(2) rigid transformation.
#     Returns aligned trajectory.
#     """
#     # Ensure trajectories are numpy arrays
#     est = np.array(est_traj)
#     gt = np.array(gt_traj)

#     # Compute centroids
#     centroid_est = np.mean(est, axis=0)
#     centroid_gt = np.mean(gt, axis=0)

#     # Center trajectories
#     est_centered = est - centroid_est
#     gt_centered = gt - centroid_gt

#     # Compute covariance matrix
#     H = est_centered.T @ gt_centered

#     # Singular Value Decomposition (SVD)
#     U, S, Vt = np.linalg.svd(H)

#     # Compute rotation matrix (R) and handle reflection
#     R = Vt.T @ U.T
#     if np.linalg.det(R) < 0:
#         Vt[-1, :] *= -1
#         R = Vt.T @ U.T

#     # Compute translation (t)
#     t = centroid_gt - R @ centroid_est

#     # Apply transformation to entire estimated trajectory
#     aligned_est = (R @ est.T).T + t

#     return aligned_est

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
    db_loop = db.get('jan28')
    db_rosbag_path = db_loop.get('rosbag_path')

    teach_rosbag_path = db_rosbag_path.get('parking_t2')
    # repeat_rosbag_path = db_rosbag_path.get('repeat1') # dont think this is needed

    # for pose graph
    trial = 't1'
    
    pose_graph_path = db_loop.get('pose_graph_path').get('temp_new_parking_t1')
    

    # temporary fixed path
    pose_graph_path = "/home/samqiao/ASRL/vtr3/temp/test_parking2/graph"
    print("pose graph path:",pose_graph_path)

    # boolean values
    db_bool = config['bool']
    SAVE = db_bool.get('SAVE')
    PLOT = db_bool.get('PLOT')

    # save folders
    result_folder = config.get('output')
    save_folder = os.path.join(result_folder, "temp_"+trial)

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

    traj = np.empty((0, 5))

    for v, e in PriviledgedIterator(v_start):
        # seems like this is from world to vertex
        # T_v_w = v.T_v_w

        C_v_w = v.T_v_w.C_ba()
        # print(v.T_v_w.matrix())
        # print("The rotation matrix is: ", C_v_w)
        x_k = v.T_v_w.r_ba_ina()[0]
        y_k = v.T_v_w.r_ba_ina()[1]
        z_k = v.T_v_w.r_ba_ina()[2]
        t_k = v.stamp / 1e9 
        roll,pitch,yaw = rotation_matrix_to_euler_angles(C_v_w)

        yaw = wrap_angle(yaw)

        x.append(x_k[0])
        y.append(y_k[0])
        t.append(t_k)

        to_append = np.array([t_k,x_k[0], y_k[0], z_k[0],yaw])

        traj = np.vstack((traj, to_append))
    
    print("for loop done the shape of traj is ",traj.shape)

    # # now we want to process the gt

    gps_ppk_path = os.path.join(parent_folder, "localization_data/ppk/parking_t2/parking_t2.txt")
    x_gt, y_gt = read_PPK_file(gps_ppk_path)

    x_gt = x_gt - x_gt[0]
    y_gt = y_gt - y_gt[0]

    traj_gt = np.squeeze(np.dstack((x_gt, y_gt)))
    traj_estimated = resample_trajectory(traj[:, 1:3],target_length=len(traj_gt))

    print("The shape of the estimated trajectory is: ", traj_estimated.shape)
    print("The shape of the ground truth is: ", traj_gt.shape)

    aligned_gt = align_trajectories(traj_estimated, traj_gt)
    


    # aligned_traj = align_trajectories(traj_estimated, traj_gt)



    if PLOT:
        plt.figure(0)
        # print("The length of x is ", len(x))
        plt.plot(traj_estimated[:,0], traj_estimated[:,1], label="Teach", linewidth=5)
        plt.plot(aligned_gt[:,0], aligned_gt[:,1], label="Ground Truth", linewidth=5)
        plt.axis('equal')
        plt.grid()
        plt.title('Teach vs Ground Truth', fontsize=20)
        plt.xlabel('x (m)', fontsize=20)
        plt.ylabel('y (m)', fontsize=20)
        plt.legend(fontsize=20)

        plt.show()

    if SAVE:
        traj_folder = os.path.join(save_folder, "traj")
        if not os.path.exists(traj_folder):
            os.makedirs(traj_folder)
            print(f"Folder '{traj_folder}' created.")

        np.savetxt(os.path.join(traj_folder, "teach_traj_estimated.txt"), traj, delimiter=",")
        # np.savetxt(os.path.join(traj_folder, "teach_traj_gt.txt"), traj_gt, delimiter=",")
        print("Teach trajectory saved.")

        # plot 3d trajectory
        plot_3d_traj(os.path.join(traj_folder, "teach_traj_estimated.txt"))
        

    print("Done.")
