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

config = load_config(os.path.join(parent_folder,'scripts/direct/direct_configs/direct_config_sam.yaml'))


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

# lets return a transformation object in SE(3) 
from pylgmath import Transformation
from pylgmath import Rotation
def parse_dro_odometry_trajectory(path_to_odometry_result):
# Read data from file
    data = []
    with open(path_to_odometry_result, 'r') as file:  # Replace with your filename
        for line in file:
            parts = line.strip().split()
            # Convert all parts to floats (ignore timestamp for plotting)
            row = list(map(float, parts))
            data.append(row)

    # Extract translation vectors (last column of each 3x4 matrix)
    T_odom = []
    timestamps = [row[0]/1e6 for row in data]  # Extract timestamps if needed

    for row in data:

        # print(row)

        # The matrix elements are indices 1 to 12 (after timestamp)
        matrix_flat = row[1:13]
        # Extract translation components: indices 3, 7, 11 in the flattened array
        a11 = matrix_flat[0]
        a12 = matrix_flat[1]
        a13 = matrix_flat[2]
        a14 = matrix_flat[3]
        a21 = matrix_flat[4]
        a22 = matrix_flat[5]
        a23 = matrix_flat[6]
        a24 = matrix_flat[7]
        a31 = matrix_flat[8]
        a32 = matrix_flat[9]
        a33 = matrix_flat[10]
        a34 = matrix_flat[11]


        C_odom = np.array([[a11, a12, a13],
                           [a21, a22, a23],
                           [a31, a32, a33]]) # Reshape to a 3x3 matrix
        
        # C_odom = C_odom.inverse()  # Invert the rotation matrix to get the correct orientation

        # print("C_odom matrix:", C_odom)
        
        r_ba_ina = np.array([[a14],
                             [a24],
                             [a34]])  # Reshape to a column vector
        
        r_ab_in_a = -C_odom.T @ r_ba_ina  # Compute r_ab_in_a
        
        # print("r_ba_ina shape:", r_ba_ina.shape)
        # print("r_ba_ina:", r_ba_ina)

        # print("r_ab_in_a shape:", r_ab_in_a.shape)

        # print("sam, ",C_odom.matrix()[0,0])

        T_odom.append(Transformation(T_ba=np.array([[a11,a12, a13, r_ab_in_a[0,0]],
                                                 [a21, a22, a23, r_ab_in_a[1,0]],
                                                 [a31,a32,a33, r_ab_in_a[2,0]],
                                                 [0, 0, 0, 1]])))

    # Convert to numpy array for easier manipulation
    T_odom = np.array(T_odom).reshape(-1,1)
    return np.array(timestamps), T_odom


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

errorx_direct = result_df['errorx_direct']
errory_direct = result_df['errory_direct']

errorx_vtr = result_df['errorx_vtr']
errory_vtr = result_df['errory_vtr']

print("gps_teach_pose", gps_teach_pose.shape)


# need to load the dro toy result as well
dro_toy_df = np.load(os.path.join(RESULT_FOLDER, f"dro_toy.npz"),allow_pickle=True)
dro_se2_pose = dro_toy_df['dro_se2_pose']
repeat_scan_stamps = dro_toy_df['repeat_scan_stamps']
timestamp_association = dro_toy_df['timestamp_association']


# I want to plot rn vtr vs direct in 3 by 1
# plot the vtr and direct results
plt.figure(figsize=(15, 5))
plt.subplot(3, 1, 1)
plt.plot(repeat_scan_stamps, dro_se2_pose[:, 0], label='DRO X Position')
plt.plot(repeat_times, vtr_se2_pose[:, 0], label='VTR X Position')
plt.plot(repeat_times, direct_se2_pose[:, 0], label='Direct X Position')
plt.xlabel('Timestamp')
plt.ylabel('X Position (m)')
plt.title('Direct SE2 Pose - X Position')
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(repeat_scan_stamps, dro_se2_pose[:, 1], label='DRO Y Position')
plt.plot(repeat_times, vtr_se2_pose[:, 1], label='VTR Y Position')
plt.plot(repeat_times, direct_se2_pose[:, 1], label='Direct Y Position')
plt.ylabel('Y Position (m)')
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(repeat_scan_stamps, dro_se2_pose[:, 2], label='DRO Yaw')
plt.plot(repeat_times, vtr_se2_pose[:, 2], label='VTR Yaw')
plt.plot(repeat_times, direct_se2_pose[:, 2], label='Direct Yaw')
plt.title('Direct SE2 Pose - Yaw')
plt.xlabel('Timestamp')
plt.ylabel('Yaw (rad)')
plt.grid()  

plt.legend()
plt.tight_layout()

plt.show()



# this plots the GT gps pose
plotter = Plotter()
plotter.plot_traj(gps_teach_pose[:,1:],gps_repeat_pose[:,1:])
plotter.show_plots()


# as well as dro odometry results (I will deal with that later)
# now we are doing it
print("sequence path:", sequence_path)
grassy_t2_dro_result = os.path.join(sequence_path, "dro_odometry_result", "grassy_t2.txt")
if not os.path.exists(grassy_t2_dro_result):
    print("ERROR: No DRO odometry result found in " + grassy_t2_dro_result)
    exit(0)
grassy_t3_dro_result = os.path.join(sequence_path, "dro_odometry_result", "grassy_t3.txt")
if not os.path.exists(grassy_t3_dro_result):
    print("ERROR: No DRO odometry result found in " + grassy_t3_dro_result)
    exit(0)

dro_t2_timestamps, dro_t2_estimates = parse_dro_odometry_trajectory(grassy_t2_dro_result)
dro_t3_timestamps, dro_t3_estimates = parse_dro_odometry_trajectory(grassy_t3_dro_result)

# x_temp = []
# y_temp = []
# for T_ba in dro_t2_estimates:
#     T_ba = T_ba[0]
#     # print("sam: this is the transformation matrix for t_ba in dro_t2_estimates: \n", T_ba.matrix())
#     x_temp.append(T_ba.r_ba_ina()[0]) # radar world in world  T_world_radar the inverse is T_radar_world   ----- r_ab_inb works which means b is the world frame and a is the radar frame
#     y_temp.append(T_ba.r_ba_ina()[1])

# x_temp_2 = []
# y_temp_2 = []
# for T_ba in dro_t3_estimates:
#     T_ba = T_ba[0]
#     print("sam: this is the transformation matrix for t_ba in dro_t3_estimates: \n", T_ba.matrix())
#     x_temp_2.append(T_ba.r_ba_ina()[0]) # radar world in world  T_world_radar the inverse is T_radar_world   ----- r_ab_inb works which means b is the world frame and a is the radar frame
#     y_temp_2.append(T_ba.r_ba_ina()[1])



# # plot it in 2d
# plt.figure(figsize=(10, 5))
# # plt.subplot(1, 2, 1)
# plt.plot(x_temp, y_temp, label='DRO T2 Estimates', color='blue')
# # plt.plot(x_temp_2, y_temp_2, label='DRO T3 Estimates', color='orange')
# # plt.plot(dro_t3_estimates[:, 0], dro_t3_estimates[:, 1], label='DRO T3 Estimates', color='orange')
# plt.axis('equal')
# plt.title('DRO Odometry Estimates')
# plt.xlabel('X Position (m)')
# plt.ylabel('Y Position (m)')
# plt.grid()
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(dro_t3_estimates[:, 0], dro_t3_estimates[:, 1], label='DRO T3 Estimates', color='orange')

# plt.axis('equal')
# plt.title('DRO Odometry Estimates')
# plt.xlabel('X Position (m)')
# plt.ylabel('Y Position (m)')
# plt.grid()
# plt.legend()

# plt.show()


r_teach_world = []
r_repeat_world_direct = [] # this is for the direct estimate
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

    # direct result we can actually do everything in SE(3) this is for the direct estimate
    r_repeat_teach_in_teach_se2 = direct_se2_pose[idx]
    # print("sam: this is direct estimate in se2: ",r_repeat_teach_in_teach_se2)


    # extend it for dro
    r_repeat_teach_in_teach_dro = dro_se2_pose[idx]
    # print("sam: this is dro estimate in se2: ",r_repeat_teach_in_teach_dro)


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

    # this chunk is for the direct estimate
    T_r_w = T_teach_repeat_direct.inverse() @ T_radar_robot @ T_teach_world # here there might a frame issue TODO

    T_gps_w_in_w_repeat = T_novatel_robot @ T_radar_robot.inverse() @ T_r_w # here there might be a frame issue TODO I think this is correct
    # r_r_w_in_world = T_r_w.r_ba_ina().T

    r_gps_w_in_w_repeat = T_gps_w_in_w_repeat.r_ba_ina() # where the gps is in the world
    # print("sam: direct r_gps_w_in_w_repeat: \n", r_gps_w_in_w_repeat)

    # print("r_r_w_in_world shape:", r_r_w_in_world.shape)
    # print("r_r_w_in_world:", r_r_w_in_world.T[0:2])

    # print("double check:", r_gps_w_in_w_repeat[0:2].shape)
    r_repeat_world_direct.append(r_gps_w_in_w_repeat[0:2].T)

    # below is for the vtr estimate
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
repeat_world_direct = np.array(r_repeat_world_direct).reshape(-1,2)
repeat_world_vtr = np.array(r_repeat_world_vtr).reshape(-1,3)

print("teach_world shape:", teach_world.shape)
print("repeat_world_direct shape:", repeat_world_direct.shape)
print("repeat_world_vtr shape:", repeat_world_vtr.shape)

# plt.figure(figsize=(10, 5))
# plt.plot(teach_world[:, 0], teach_world[:, 1], label='VTR Teach World', color='blue')
# plt.plot(repeat_world_direct[:, 0], repeat_world_direct[:, 1], label='Direct Repeat World', color='orange')
# plt.plot(repeat_world_vtr[:, 0], repeat_world_vtr[:, 1], label='VTR Repeat World', color='green')
# plt.axis('equal')
# plt.title('Teach and Repeat World Trajectories')
# plt.xlabel('X Position (m)')
# plt.ylabel('Y Position (m)')
# plt.grid()
# plt.legend()
# plt.show()


# we need to keep error calculation the same as the vtr one and previous direct one, the only difference here is the shape
# actually dro is quite different but the core logic should be the same
print("dro_se2_pose shape:", dro_se2_pose.shape)
print("timestamp_association shape:", timestamp_association.shape)


# error is calculated as the difference in x and y in the odomtry teach world frame for the estimated repeat pose vs the ground truth repeat pose all expressed in the odometry teach world frame

# first we need to get SE(3) transformation of the teach odom 

r_teach_world_dro = []
r_repeat_world_dro = [] # this is for the dro estimate

for dro_odomtry_idx in range(0, dro_se2_pose.shape[0]):
    
    # for the teach pose
    T_radar_world_in_world_dro = dro_t2_estimates[dro_odomtry_idx][0] # this is the transformation from the radar to the world in the teach odometry frame 

    T_robot_world_in_world_dro = T_radar_robot.inverse() @ T_radar_world_in_world_dro # this is the transformation from the robot to the world in the teach odometry frame
    T_gps_world_in_world_dro = T_novatel_robot @ T_robot_world_in_world_dro # this is the transformation from the gps to the world in the teach odometry frame

    # now there is no correspondence here
    r_repeat_teach_in_teach_se2_dro = dro_se2_pose[dro_odomtry_idx]
    # print("sam: this is direct estimate in se2: ",r_repeat_teach_in_teach_se2)
    T_teach_repeat_dro = Transformation(T_ba = se2_to_se3(r_repeat_teach_in_teach_se2_dro))



    # we can recover the repeat pose in the teach odometry frame
     # this chunk is for the direct estimate # this is not true need to use the assiociation to find it: T_robot_world_in_world_dro
    # lets try....
    association_at_idx = timestamp_association[dro_odomtry_idx][0]
    teach_time_dro = float(list(association_at_idx.values())[0])
    repeat_time_dro = float(list(association_at_idx.keys())[0])


    # so! I need to what the pose of the teach is at the time of the repeat (use timestamp to find the closest one)
    dro_teach_idx = np.argmin(np.abs(dro_t2_timestamps - teach_time_dro))

    T_radar_world_in_world_dro_corresponding_to_map = dro_t2_estimates[dro_teach_idx][0] # this is the transformation from the radar to the world in the teach odometry frame at the time of the repeat
    T_robot_world_in_world_dro_corresponding_to_map = T_radar_robot.inverse() @ T_radar_world_in_world_dro_corresponding_to_map # this is the transformation from the robot to the world in the teach odometry frame at the time of the repeat
    T_gps_world_in_world_dro_corresponding_to_map = T_novatel_robot @ T_robot_world_in_world_dro_corresponding_to_map # this is the transformation from the gps to the world in the teach odometry frame at the time of the repeat

    r_teach_world_dro.append(T_gps_world_in_world_dro_corresponding_to_map.r_ba_ina()[0:2].T) # where the gps is in the world in the teach odometry frame


    # for repeat pose
    T_r_w = T_teach_repeat_dro.inverse() @ T_radar_world_in_world_dro_corresponding_to_map # here there might a frame issue TODO

    T_gps_w_in_w_repeat = T_novatel_robot @ T_radar_robot.inverse() @ T_r_w # here there might be a frame issue TODO I think this is correct
    # r_r_w_in_world = T_r_w.r_ba_ina().T

    r_gps_w_in_w_repeat = T_gps_w_in_w_repeat.r_ba_ina() # where the gps is in the world

    r_repeat_world_dro.append(r_gps_w_in_w_repeat[0:2].T)


# lets plot the results teach and repeat
r_teach_world_dro = np.array(r_teach_world_dro).reshape(-1,2)
r_repeat_world_dro = np.array(r_repeat_world_dro).reshape(-1,2)
print("r_repeat_world_dro shape:", r_repeat_world_dro.shape)
print("r_teach_world_dro shape:", r_teach_world_dro.shape)

plt.figure(figsize=(10, 5))
plt.scatter(r_teach_world_dro[:, 0], r_teach_world_dro[:, 1], label='DRO Teach World', color='blue')
plt.scatter(r_repeat_world_dro[:, 0], r_repeat_world_dro[:, 1], label='DRO Repeat World', color='orange')
plt.axis('equal')
plt.title('DRO Odometry Estimates')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.grid()
plt.legend()
plt.show()

# # lets see the first 10 elements of repeat dro
# print("r_repeat_world_dro first 10 elements:", r_repeat_world_dro[:10,:])

# now we have to do some hacky align trajectory logic as before
window_size = 50

errorx_dro = []
errory_dro = []

# first window size
def get_piecewise_path_length(gt_trajectory):
    pose = gt_trajectory[:,1:] # this is a 3 by 1 

    length = np.sum(np.sqrt(np.sum(np.diff(pose, axis=0)**2, axis=1)))

    return length
# the first window size points we use the future window size points 20 points lets say
for dro_repeat_idx in range(0,window_size):
    print("--------------Processing repeat_idx:", dro_repeat_idx ,"-----------------")
    corr_gps_teach = []
    corr_gps_repeat = []
    for window_idx in range(0,window_size):

        # the get time is different: we use the association of the timestamps
        association_at_idx = timestamp_association[dro_repeat_idx + window_idx][0]

        # print("association_at_idx:", association_at_idx)

        teach_time_dro = float(list(association_at_idx.values())[0])
        repeat_time_dro = float(list(association_at_idx.keys())[0])

        # print("teach_time:", teach_time_dro)
        # print("repeat_time:", repeat_time_dro)


        # get the gps pose at the time (time correspondence)
        corr_gps_pose_teach = gps_teach_pose[np.argmin(np.abs(gps_teach_pose[:,0] - teach_time_dro)),:]
        corr_gps_pose_repeat = gps_repeat_pose[np.argmin(np.abs(gps_repeat_pose[:,0] - repeat_time_dro)),:]

        corr_gps_teach.append(corr_gps_pose_teach)
        corr_gps_repeat.append(corr_gps_pose_repeat)
  
    
    corr_gps_teach = np.array(corr_gps_teach).reshape(-1,4)
    corr_gps_repeat = np.array(corr_gps_repeat).reshape(-1,4)

    segment_length = get_piecewise_path_length(corr_gps_teach)
    print("the segment teach length is: ", segment_length,"m")
    if segment_length < 5:
        print("segment length is too small!")
        # continue
        # raise ValueError("segment length is too small!")


    # now we do the alignment for teach
    window_estimated_teach = r_teach_world_dro[dro_repeat_idx:dro_repeat_idx+window_size,:2]

    window_estimated_repeat_dro = r_repeat_world_dro[dro_repeat_idx:dro_repeat_idx+window_size,:2]
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

    repeat_estimated = window_estimated_repeat_dro[0]

    print("repeat_ppk_at_idx:", repeat_ppk_at_idx)
    print("repeat_estimated dro:", repeat_estimated)

    error = repeat_estimated - repeat_ppk_at_idx

    print("error :", error)

    errorx_dro.append(error[0])
    errory_dro.append(error[1])


# we do that for the next window case
# this takes care of window size to N-window size points
for dro_repeat_idx in range(window_size,dro_se2_pose.shape[0]-window_size,1):
    print("--------------Processing repeat_idx:", dro_repeat_idx ,"-----------------")
    if(dro_repeat_idx + window_size) > dro_se2_pose.shape[0]:
        break # exit when it is out of bounds
    corr_gps_teach = []
    corr_gps_repeat = []

    half_window_size = int(window_size/2)
    for window_idx in range(-half_window_size,half_window_size,1):
        # the get time is different: we use the association of the timestamps
        association_at_idx = timestamp_association[dro_repeat_idx + window_idx][0]

        # print("association_at_idx:", association_at_idx)

        teach_time_dro = float(list(association_at_idx.values())[0])
        repeat_time_dro = float(list(association_at_idx.keys())[0])

        # print("teach_time:", teach_time_dro)
        # print("repeat_time:", repeat_time_dro)


        # get the gps pose at the time (time correspondence)
        corr_gps_pose_teach = gps_teach_pose[np.argmin(np.abs(gps_teach_pose[:,0] - teach_time_dro)),:]
        corr_gps_pose_repeat = gps_repeat_pose[np.argmin(np.abs(gps_repeat_pose[:,0] - repeat_time_dro)),:]

        corr_gps_teach.append(corr_gps_pose_teach)
        corr_gps_repeat.append(corr_gps_pose_repeat)
    
    corr_gps_teach = np.array(corr_gps_teach).reshape(-1,4)
    corr_gps_repeat = np.array(corr_gps_repeat).reshape(-1,4)

    segment_length = get_piecewise_path_length(corr_gps_teach)
    print("the segment teach length is: ", segment_length,"m")
    if segment_length < 5:
        print("segment length is too small!")
        # continue
        raise ValueError("segment length is too small!")
    
    # now we do the alignment for teach
        # now we do the alignment for teach (half window size)
    window_estimated_teach = r_teach_world_dro[dro_repeat_idx-half_window_size:dro_repeat_idx+half_window_size,:2]
    window_estimated_repeat_dro = r_repeat_world_dro[dro_repeat_idx-half_window_size:dro_repeat_idx+half_window_size,:2]
    
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
    repeat_ppk_at_idx = repeat_ppk_in_odom[half_window_size]
    repeat_estimated = window_estimated_repeat_dro[half_window_size]
    print("repeat_ppk_at_idx:", repeat_ppk_at_idx)
    print("repeat_estimated dro:", repeat_estimated)

    error = repeat_estimated - repeat_ppk_at_idx
    errorx_dro.append(error[0])
    errory_dro.append(error[1])

    print("error :", error)

# the last window size to the end
for dro_repeat_idx in range(dro_se2_pose.shape[0]-window_size,dro_se2_pose.shape[0]):
    print("--------------Processing repeat_idx:", dro_repeat_idx ,"-----------------")
    # if(repeat_idx - window_size) > repeat_times.shape[0]:
    #     break # exit when it is out of bounds
    corr_gps_teach = []
    corr_gps_repeat = []

     # use the past window size points
    for window_idx in range(-window_size,0,1):

        # the get time is different: we use the association of the timestamps
        association_at_idx = timestamp_association[dro_repeat_idx + window_idx][0]

        # print("association_at_idx:", association_at_idx)

        teach_time_dro = float(list(association_at_idx.values())[0])
        repeat_time_dro = float(list(association_at_idx.keys())[0])

        # get the gps pose at the time
        corr_gps_pose_teach = gps_teach_pose[np.argmin(np.abs(gps_teach_pose[:,0] - teach_time_dro)),:]
        corr_gps_pose_repeat = gps_repeat_pose[np.argmin(np.abs(gps_repeat_pose[:,0] - repeat_time_dro)),:]

        corr_gps_teach.append(corr_gps_pose_teach)
        corr_gps_repeat.append(corr_gps_pose_repeat)
    
    corr_gps_teach = np.array(corr_gps_teach).reshape(-1,4)
    corr_gps_repeat = np.array(corr_gps_repeat).reshape(-1,4)

    segment_length = get_piecewise_path_length(corr_gps_teach)
    print("the segment teach length is: ", segment_length,"m")
    if segment_length < 10:
        print("segment length is too small!")
        # continue
        # raise ValueError("segment length is too small!")

    # print("corr_gps_teach shape:", corr_gps_teach.shape)

    # now we do the alignment for teach
    window_estimated_teach = r_teach_world_dro[dro_repeat_idx-window_size:dro_repeat_idx,:2]
    window_estimated_repeat_dro = r_repeat_world_dro[dro_repeat_idx-window_size:dro_repeat_idx,:2]

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
    repeat_estimated = window_estimated_repeat_dro[window_size-1]


    error = repeat_estimated - repeat_ppk_at_idx

    errorx_dro.append(error[0])
    errory_dro.append(error[1])
    print("repeat_ppk_at_idx:", repeat_ppk_at_idx)
    print("repeat_estimated dro:", repeat_estimated)
    print("error :", error)


# now we have the error in x and y
errorx_dro = np.array(errorx_dro).reshape(-1,1)
errory_dro = np.array(errory_dro).reshape(-1,1)

print("errorx_dro shape:", errorx_dro.shape)
print("errory_dro shape:", errory_dro.shape)
 

errorx_dro[np.abs(errorx_dro)>1]=0
errory_dro[np.abs(errory_dro)>1]=0

 # can we plot the error? in 2 by 1 plot
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(dro_t3_timestamps[1:],errorx_dro, label='DRO X Error', color='blue')
plt.plot(repeat_times, errorx_direct, label='Direct X Error', color='green')
plt.plot(repeat_times, errorx_vtr, label='VTR X Error', color='red')
plt.title('Error in X (m)')
plt.xlabel('timestamp')
plt.ylabel('X Error (m)')
plt.grid()
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(dro_t3_timestamps[1:], errory_dro, label='DRO Y Error', color='orange')
plt.plot(repeat_times, errory_direct, label='Direct Y Error', color='green')
plt.plot(repeat_times, errory_vtr, label='VTR Y Error', color='red')
plt.title('Error in Y (m)')
plt.xlabel('timestamp')
plt.ylabel('Y Error (m)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

