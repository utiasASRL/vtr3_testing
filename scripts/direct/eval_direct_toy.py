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
parent_folder = "/home/sahakhsh/Documents/vtr3_testing"

# Insert path at index 0 so it's searched first
sys.path.insert(0, parent_folder)

# from deps.path_tracking_error.fcns import *

# from radar.utils.helper import *

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

config = load_config(os.path.join(parent_folder,'scripts/direct/direct_config_hshmat.yaml'))


db_bool = config['bool']
SAVE = db_bool.get('SAVE')
SAVE = False
print("SAVE:",SAVE)
PLOT = db_bool.get('PLOT')
DEBUG = db_bool.get('DEBUG')

result_folder = config.get('output')

# change here
out_path_folder = os.path.join(result_folder,f"grassy_t2_r3/")
if not os.path.exists(out_path_folder):
    os.makedirs(out_path_folder)
    print(f"Folder '{out_path_folder}' created.")
else:
    print(f"Folder '{out_path_folder}' already exists.")    



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

    # # # Ensure a proper rotation matrix (det(R) should be 1)
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

    return gt_aligned, R, t


# print(result_folder)

sequence = "grassy_t2_r3"

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

print("gps_teach_pose", gps_teach_pose.shape)

plotter = Plotter()
plotter.plot_traj(gps_teach_pose[:,1:],gps_repeat_pose[:,1:])
plotter.show_plots()


r_teach_world = []
r_repeat_world = []
r_repeat_world_vtr = []
for idx in range(0,vtr_se2_pose.shape[0]):
    # the time stamps
    teach_vertex_time = teach_times[idx]
    repeat_vertex_time = repeat_times[idx]

    T_teach_world = teach_vertex_transforms[idx][0][teach_vertex_time[0]]
    T_gps_world_teach = T_novatel_robot @ T_teach_world
    r_gps_w_in_w_teach = T_gps_world_teach.r_ba_ina() # where the gps is in the world
    # r_teach_world_in_world = T_teach_world.r_ba_ina()
    r_teach_world.append(r_gps_w_in_w_teach.T)

    # direct result we can actually do everything in SE(3)
    r_repeat_teach_in_teach_se2 = direct_se2_pose[idx]
    # print("sam: this is direct estimate in se2: ",r_repeat_teach_in_teach_se2)

    # ok lets define the rotation and translation vector and then use the transformation matrix class
    def se2_to_se3(se2_vec):
        """
        Convert SE(2) pose to SE(3) transformation matrix
        
        Args:
            se2_vec: Array-like of shape (3,) containing [x, y, theta] 
                    (theta in radians)
        
        Returns:
            4x4 SE(3) transformation matrix as NumPy array
        """
        # Ensure input is flattened and extract components
        x, y, theta = np.asarray(se2_vec).flatten()
        
        # Create rotation matrix (Z-axis rotation)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([
            [c, -s,  0],
            [s,  c,  0],
            [0,  0,  1]
        ])
        
        # Create translation vector
        translation = np.array([x, y, 0])
        
        # Construct SE(3) matrix
        se3_matrix = np.eye(4)
        se3_matrix[:3, :3] = R          # Set rotation
        se3_matrix[:3, 3] = translation # Set translation
        
        return se3_matrix
    
    T_teach_repeat_direct = Transformation(T_ba = se2_to_se3(r_repeat_teach_in_teach_se2))
    # print("sam: this is direct estimate in se(3): \n",T_teach_repeat_direct.matrix())


    T_r_w = T_teach_repeat_direct.inverse() @ T_radar_robot @ T_teach_world # here there might a frame issue TODO

    T_gps_w_in_w_repeat = T_novatel_robot @ T_radar_robot.inverse() @ T_r_w # here there might be a frame issue TODO I think this is correct
    # r_r_w_in_world = T_r_w.r_ba_ina().T

    r_gps_w_in_w_repeat = T_gps_w_in_w_repeat.r_ba_ina() # where the gps is in the world
    # print("sam: direct r_gps_w_in_w_repeat: \n", r_gps_w_in_w_repeat)

    # print("r_r_w_in_world shape:", r_r_w_in_world.shape)
    # print("r_r_w_in_world:", r_r_w_in_world.T[0:2])

    # print("double check:", r_gps_w_in_w_repeat[0:2].shape)
    r_repeat_world.append(r_gps_w_in_w_repeat[0:2].T)

    # need to investigate vtr_repeat (TODO) seems to be on the other side of the teach compared to direct 
    T_teach_repeat_edge = repeat_edge_transforms[idx][0][repeat_vertex_time[0]]

    # print("sam: this is vtr estimate se(3): \n",T_teach_repeat_edge.matrix())

    T_repeat_w = T_teach_repeat_edge.inverse() @ T_teach_world
    T_gps_w_in_w_repeat_vtr = T_novatel_robot @ T_repeat_w
    r_gps_w_in_w_repeat_vtr = T_gps_w_in_w_repeat_vtr.r_ba_ina()
    

    r_repeat_world_vtr.append(r_gps_w_in_w_repeat_vtr.T)
    # print("sam: vtr r_gps_w_in_w_repeat: \n", r_gps_w_in_w_repeat_vtr)

# make them into numpy arrays
teach_world = np.array(r_teach_world).reshape(-1,3)
repeat_world_direct = np.array(r_repeat_world).reshape(-1,2)
repeat_world_vtr = np.array(r_repeat_world_vtr).reshape(-1,3)

print("teach_world shape:", teach_world.shape)
print("repeat_world_direct shape:", repeat_world_direct.shape)
print("repeat_world_vtr shape:", repeat_world_vtr.shape)

# plotter.plot_traj(teach_world,repeat_world_direct)
# plotter.show_plots()

window_size = 50

errorx_direct = []
errory_direct = []

errorx_vtr = []
errory_vtr = []
# errortheta = [] later maybe we can use the tangent....


def get_piecewise_path_length(gt_trajectory):
    pose = gt_trajectory[:,1:] # this is a 3 by 1 

    length = np.sum(np.sqrt(np.sum(np.diff(pose, axis=0)**2, axis=1)))

    return length
# the first window size points we use the future window size points 20 points lets say
for repeat_idx in range(0,window_size):
    print("--------------Processing repeat_idx:", repeat_idx ,"-----------------")
    corr_gps_teach = []
    corr_gps_repeat = []
    for window_idx in range(0,window_size):
        teach_time = teach_times[repeat_idx+window_idx]
        repeat_time = repeat_times[repeat_idx+window_idx]

        # get the gps pose at the time (time correspondence)
        corr_gps_pose_teach = gps_teach_pose[np.argmin(np.abs(gps_teach_pose[:,0] - teach_time[0])),:]
        corr_gps_pose_repeat = gps_repeat_pose[np.argmin(np.abs(gps_repeat_pose[:,0] - repeat_time[0])),:]

        corr_gps_teach.append(corr_gps_pose_teach)
        corr_gps_repeat.append(corr_gps_pose_repeat)
  
    
    corr_gps_teach = np.array(corr_gps_teach).reshape(-1,4)
    corr_gps_repeat = np.array(corr_gps_repeat).reshape(-1,4)

    segment_length = get_piecewise_path_length(corr_gps_teach)
    print("the segment teach length is: ", segment_length,"m")
    if segment_length < 10:
        print("segment length is too small!")
        # raise ValueError("segment length is too small!")


    # now we do the alignment for teach
    window_estimated_teach = teach_world[repeat_idx:repeat_idx+window_size,:2]
    window_estimated_repeat_direct = repeat_world_direct[repeat_idx:repeat_idx+window_size,:2]
    aligned_teach_ppk_in_odom, R_teach_ppk, t_teach_ppk = align_trajectories(window_estimated_teach,corr_gps_teach[:,1:3]) # align x,y to x,y

    # we transform the repeat gps pose to the odom frame
    repeat_ppk_in_odom = []
    for idx in range(0,window_size):
        gt_repeat = corr_gps_repeat[idx, 1:3]

        gt_repeat = np.dot(gt_repeat, R_teach_ppk)
        gt_repeat += t_teach_ppk
        repeat_ppk_in_odom.append(gt_repeat)

    repeat_ppk_in_odom = np.array(repeat_ppk_in_odom).reshape(-1,2)

    # they are all expressed in the odom frame
    repeat_ppk_at_idx = repeat_ppk_in_odom[0]

    repeat_estimated = window_estimated_repeat_direct[0]

    print("repeat_ppk_at_idx:", repeat_ppk_at_idx)
    print("repeat_estimated direct:", repeat_estimated)
    print("repeat_estimated vtr:", repeat_world_vtr[repeat_idx,:2])

    error = repeat_estimated - repeat_ppk_at_idx

    errorx_direct.append(error[0])
    errory_direct.append(error[1])

    # error vtr
    r_repeat_teach_in_teach = repeat_world_vtr[repeat_idx,:2]
    error_vtr = r_repeat_teach_in_teach - repeat_ppk_at_idx
    errorx_vtr.append(error_vtr[0])
    errory_vtr.append(error_vtr[1])
    
    # break

# exit(0)
# this takes care of window size to N-window size points
for repeat_idx in range(window_size,repeat_times.shape[0]-window_size,1):
    print("--------------Processing repeat_idx:", repeat_idx ,"-----------------")
    if(repeat_idx + window_size) > repeat_times.shape[0]:
        break # exit when it is out of bounds
    corr_gps_teach = []
    corr_gps_repeat = []

    half_window_size = int(window_size/2)
    for window_idx in range(-half_window_size,half_window_size,1):
        # print("sam : window_idx:", window_idx)
        teach_time = teach_times[repeat_idx+window_idx]
        repeat_time = repeat_times[repeat_idx+window_idx]

        # get the gps pose at the time
        corr_gps_pose_teach = gps_teach_pose[np.argmin(np.abs(gps_teach_pose[:,0] - teach_time[0])),:]
        corr_gps_pose_repeat = gps_repeat_pose[np.argmin(np.abs(gps_repeat_pose[:,0] - repeat_time[0])),:]

        corr_gps_teach.append(corr_gps_pose_teach)
        corr_gps_repeat.append(corr_gps_pose_repeat)
  
    
    corr_gps_teach = np.array(corr_gps_teach).reshape(-1,4)
    corr_gps_repeat = np.array(corr_gps_repeat).reshape(-1,4)

    segment_length = get_piecewise_path_length(corr_gps_teach)
    print("the segment teach length is: ", segment_length,"m")
    if segment_length < 10:
        print("segment length is too small!")
     
        continue
    
    # now we do the alignment for teach (half window size)
    window_estimated_teach = teach_world[repeat_idx-half_window_size:repeat_idx+half_window_size,:2]
    window_estimated_repeat_direct = repeat_world_direct[repeat_idx-half_window_size:repeat_idx+half_window_size,:2]
    
    aligned_teach_ppk_in_odom, R_teach_ppk, t_teach_ppk = align_trajectories(window_estimated_teach,corr_gps_teach[:,1:3]) # align x,y to x,y

    # we transform the repeat gps pose to the odom frame
    repeat_ppk_in_odom = []
    for idx in range(0,window_size):
        gt_repeat = corr_gps_repeat[idx, 1:3]

        gt_repeat = np.dot(gt_repeat, R_teach_ppk)
        gt_repeat += t_teach_ppk
        repeat_ppk_in_odom.append(gt_repeat)

    repeat_ppk_in_odom = np.array(repeat_ppk_in_odom).reshape(-1,2)


    repeat_ppk_at_idx = repeat_ppk_in_odom[half_window_size]

    repeat_estimated = window_estimated_repeat_direct[half_window_size]

    print("repeat_ppk_at_idx:", repeat_ppk_at_idx)
    print("repeat_estimated:", repeat_estimated)
    print("repeat_estimated vtr:", repeat_world_vtr[repeat_idx,:2])

    error = repeat_estimated - repeat_ppk_at_idx

    errorx_direct.append(error[0])
    errory_direct.append(error[1])

    # error vtr
    r_repeat_teach_in_teach = repeat_world_vtr[repeat_idx,:2]
    error_vtr = r_repeat_teach_in_teach - repeat_ppk_at_idx
    errorx_vtr.append(error_vtr[0])
    errory_vtr.append(error_vtr[1])


    # break


# starting 1396
# the last window size to the end
for repeat_idx in range(repeat_times.shape[0]-window_size,repeat_times.shape[0]):
    print("--------------Processing repeat_idx:", repeat_idx ,"-----------------")
    # if(repeat_idx - window_size) > repeat_times.shape[0]:
    #     break # exit when it is out of bounds
    corr_gps_teach = []
    corr_gps_repeat = []

    # use the past window size points
    for window_idx in range(-window_size,0,1):
        # print("sam : window_idx:", window_idx)
        teach_time = teach_times[repeat_idx+window_idx]
        repeat_time = repeat_times[repeat_idx+window_idx]

        # get the gps pose at the time
        corr_gps_pose_teach = gps_teach_pose[np.argmin(np.abs(gps_teach_pose[:,0] - teach_time[0])),:]
        corr_gps_pose_repeat = gps_repeat_pose[np.argmin(np.abs(gps_repeat_pose[:,0] - repeat_time[0])),:]

        corr_gps_teach.append(corr_gps_pose_teach)
        corr_gps_repeat.append(corr_gps_pose_repeat)
    
    corr_gps_teach = np.array(corr_gps_teach).reshape(-1,4)
    corr_gps_repeat = np.array(corr_gps_repeat).reshape(-1,4)

    segment_length = get_piecewise_path_length(corr_gps_teach)
    print("the segment teach length is: ", segment_length,"m")
    if segment_length < 10:
        print("segment length is too small!")
        # raise ValueError("segment length is too small!")

    # print("corr_gps_teach shape:", corr_gps_teach.shape)

    # now we do the alignment for teach
    window_estimated_teach = teach_world[repeat_idx-window_size:repeat_idx,:2]
    window_estimated_repeat_direct = repeat_world_direct[repeat_idx-window_size:repeat_idx,:2]

    aligned_teach_ppk_in_odom, R_teach_ppk, t_teach_ppk = align_trajectories(window_estimated_teach,corr_gps_teach[:,1:3]) # align x,y to x,y
    # we transform the repeat gps pose to the odom frame
    repeat_ppk_in_odom = []
    for idx in range(0,window_size):
        gt_repeat = corr_gps_repeat[idx, 1:3]

        gt_repeat = np.dot(gt_repeat, R_teach_ppk)
        gt_repeat += t_teach_ppk
        repeat_ppk_in_odom.append(gt_repeat)
    repeat_ppk_in_odom = np.array(repeat_ppk_in_odom).reshape(-1,2)
    repeat_ppk_at_idx = repeat_ppk_in_odom[window_size-1]  
    repeat_estimated = window_estimated_repeat_direct[window_size-1]

    print("repeat_ppk_at_idx:", repeat_ppk_at_idx)
    print("repeat_estimated direct:", repeat_estimated)
    print("repeat_estimated vtr:", repeat_world_vtr[repeat_idx,:2])

    error = repeat_estimated - repeat_ppk_at_idx
    errorx_direct.append(error[0])
    errory_direct.append(error[1])

    # error vtr
    r_repeat_teach_in_teach = repeat_world_vtr[repeat_idx,:2]
    error_vtr = r_repeat_teach_in_teach - repeat_ppk_at_idx
    errorx_vtr.append(error_vtr[0])
    errory_vtr.append(error_vtr[1])

    # break


errorx_direct = np.array(errorx_direct).reshape(-1,1)
errory_direct = np.array(errory_direct).reshape(-1,1)

errorx_vtr = np.array(errorx_vtr).reshape(-1,1)
errory_vtr = np.array(errory_vtr).reshape(-1,1)

error_norm_direct = np.linalg.norm(np.hstack((errorx_direct,errory_direct)),axis=1)
error_norm_vtr = np.linalg.norm(np.hstack((errorx_vtr,errory_vtr)),axis=1)

# find where the norm is the largest
max_idx = np.argmax(error_norm_direct)
print("max_idx:", max_idx)

print("errorx_direct shape:", errorx_direct.shape)
print("errory_direct shape:", errory_direct.shape)
print("error_norm shape:", error_norm_direct.shape)


starting_plot_idx = 1


plt.figure(1) # need to make it 2 by 1
plt.subplot(3, 1, 1) # I wan to have rmse in the label
plt.plot(repeat_times[starting_plot_idx:], errorx_vtr[starting_plot_idx:], label=f'RMSE x vtr: {np.sqrt(np.mean(errorx_vtr[starting_plot_idx:]**2)):.4f} m')
plt.plot(repeat_times[starting_plot_idx:], errorx_direct[starting_plot_idx:], label=f"RMSE x direct: {np.sqrt(np.mean(errorx_direct[starting_plot_idx:]**2)):.4f} m")
plt.title('Direct & VTR estimate error in x,y')
plt.ylabel('error x (m)')
plt.grid()
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(repeat_times[starting_plot_idx:], errory_vtr[starting_plot_idx:], label=f'RMSE y vtr: {np.sqrt(np.mean(errory_vtr**2)):.4f} m')
plt.plot(repeat_times[starting_plot_idx:], errory_direct[starting_plot_idx:], label=f'RMSE y direct: {np.sqrt(np.mean(errory_direct**2)):.4f} m')
plt.ylabel('error y (m)')
plt.grid()
# plt.xlabel('Repeat Times')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(repeat_times[starting_plot_idx:], error_norm_vtr[starting_plot_idx:], label='VTR error norm (Euclidean), RMSE: {:.4f} m'.format(np.sqrt(np.mean(error_norm_vtr[starting_plot_idx:]**2))))
plt.plot(repeat_times[starting_plot_idx:], error_norm_direct[starting_plot_idx:], label='Direct error norm (Euclidean), RMSE: {:.4f} m'.format(np.sqrt(np.mean(error_norm_direct[starting_plot_idx:]**2))))
plt.ylabel('error norm (m)')
plt.grid()
plt.legend()
plt.xlabel('Repeat Times')


# plotter.set_data(sequence_path)
# plotter.plot_localziation_error()
plotter.show_plots()
    


