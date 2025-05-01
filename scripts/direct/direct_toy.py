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

config = load_config(os.path.join(parent_folder,'scripts/direct/direct_config.yaml'))


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

# the ppk data wrt the gps path tracking error
ppk_data = np.load(f"{out_path_folder}repeat_path_tracking_error.npz")

# I load the ppk data path tracking error
t_repeat_ppk = ppk_data['t_repeat_ppk']
distances_teach_repeat_ppk = ppk_data['distances_teach_repeat_ppk']

if PLOT:
    plt.figure()
   
    plt.title('VTR Estimated Path Tracking Error')
    # reset t_repeat times to start from 0
    t_repeat = repeat_times - repeat_times[0]
    plt.plot(t_repeat,vtr_estimated_ptr, label=f'VTR RMSE: {np.sqrt(np.mean(vtr_estimated_ptr**2)):.3f}m for Repeat  Max Error: {np.max(np.abs(vtr_estimated_ptr)):.3f}m')
    
    # print("t_repeat_ppk shape:", t_repeat_ppk.shape)
    # print("distances_teach_repeat_ppk shape:", distances_teach_repeat_ppk.shape)
    plt.plot(t_repeat_ppk, distances_teach_repeat_ppk, label=f"PPK RMSE: {np.sqrt(np.mean(distances_teach_repeat_ppk**2)):.3f}m for Repeat  Max Error: {np.max(np.abs(distances_teach_repeat_ppk)):.3f}m")
    
    plt.xlabel('Repeat Times')
    plt.ylabel('Path Tracking Error (m)')
    plt.grid()
    plt.legend()
    plt.show()


# now we will get a baseline quick norm 
# also I need to verify the xyz in a plane
ppk = config['ppk']
teach_ppk_folder = ppk['teach']
repeat_ppk_folder = ppk['repeat']

# teach is grassy t2
# repeat is grassy t3
grassy_t2_ppk = os.path.join(teach_ppk_folder, "grassy_t2.txt")
grassy_t2_ppk_ros = os.path.join(teach_ppk_folder, "grassy_t2_ros.txt")
grassy_t3_ppk = os.path.join(repeat_ppk_folder, "grassy_t3.txt")
grassy_t3_ppk_ros = os.path.join(repeat_ppk_folder, "grassy_t3_ros.txt")

t_teach_ppk = read_gps_ros_txt(grassy_t2_ppk_ros)
t_repeat_ppk = read_gps_ros_txt(grassy_t3_ppk_ros)

x_teach_ppk, y_teach_ppk = read_PPK_file_sam(grassy_t2_ppk)
x_repeat_ppk, y_repeat_ppk = read_PPK_file_sam(grassy_t3_ppk)

print("t_teach_ppk shape:", t_teach_ppk.shape)
print("x_teach_ppk shape:", x_teach_ppk.shape)
print("y_teach_ppk shape:", y_teach_ppk.shape)

print("t_repeat_ppk shape:", t_repeat_ppk.shape)
print("x_repeat_ppk shape:", x_repeat_ppk.shape)
print("y_repeat_ppk shape:", y_repeat_ppk.shape)

# x_teach_ppk,y_teach_ppk,z_teach_ppk = read_PPK_file(grassy_t2_ppk)
# teach_ppk_length_3d = get_path_distance_from_gps_3D(x_teach_ppk,y_teach_ppk,z_teach_ppk)
teach_ppk_length_2d = get_path_distance_from_gps(x_teach_ppk,y_teach_ppk)
print("Teach PPK length in 2D:", teach_ppk_length_2d)
repeat_ppk_length_2d = get_path_distance_from_gps(x_repeat_ppk,y_repeat_ppk)
print("Repeat PPK length in 2D:", repeat_ppk_length_2d)
# print("Repeat PPK length in 3D:", repeat_ppk_length_3d)


# now it is possible to stack them t_repeat_gps, x_repeat_ppk, y_repeat_ppk
r2_pose_repeat_ppk_dirty = np.hstack((t_repeat_ppk.reshape(-1,1), x_repeat_ppk.reshape(-1,1), y_repeat_ppk.reshape(-1,1), np.zeros_like(x_repeat_ppk.reshape(-1,1))))
print("r2_pose_ppk_dirty shape:",r2_pose_repeat_ppk_dirty.shape) 

# do the same for teach
# now it is possible to stack them t_teach_gps, x_teach_ppk, y_teach_ppk
r2_pose_teach_ppk_dirty = np.hstack((t_teach_ppk.reshape(-1,1), x_teach_ppk.reshape(-1,1), y_teach_ppk.reshape(-1,1),np.zeros_like(x_teach_ppk.reshape(-1,1))))
print("r2_pose_teach_ppk_dirty shape:",r2_pose_teach_ppk_dirty.shape)

# so the plot should be x-axis is the repeat time (can start from 0) and y-axis is the SE(2) norm
# for both GPS and VTR

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

    # delta_x_delta_y_ppk = repeat_ppk[1:3] - teach_ppk[1:3]

    gps_norm.append(np.linalg.norm(teach_ppk[1:3] - repeat_ppk[1:3]))
    print("gps norm:", gps_norm[repeat_vertex_idx])
    # print("T_teach_repeat_edge_options:", T_teach_repeat_edge_options)


dir_norm = []
vtr_se2_pose = []
direct_se2_pose = []
print("------ dir norm ------")
# now we can set up the direct localization stuff
for repeat_vertex_idx in range(0,repeat_times.shape[0]):
    print("------------------ repeat idx: ", repeat_vertex_idx,"------------------")
    repeat_vertex_time = repeat_times[repeat_vertex_idx]

    teach_cv_scan_polar = teach_polar_imgs[repeat_vertex_idx]
    teach_scan_azimuth_angles = teach_azimuth_angles[repeat_vertex_idx]
    teach_scan_timestamps = teach_azimuth_timestamps[repeat_vertex_idx]

    repeat_cv_scan_polar = repeat_polar_imgs[repeat_vertex_idx]
    repeat_scan_azimuth_angles = repeat_azimuth_angles[repeat_vertex_idx]
    repeat_scan_timestamps = repeat_azimuth_timestamps[repeat_vertex_idx]
    
    class RadarFrame:
        def __init__(self, polar, azimuths, timestamps):
            self.polar = polar[:, :].astype(np.float32) / 255.0
            self.azimuths=azimuths
            self.timestamps=timestamps.flatten().astype(np.int64)  

    teach_frame = RadarFrame(teach_cv_scan_polar, teach_scan_azimuth_angles, teach_scan_timestamps.reshape(-1,1))
    repeat_frame = RadarFrame(repeat_cv_scan_polar, repeat_scan_azimuth_angles, repeat_scan_timestamps.reshape(-1,1))

    # we can use teach and repeat result as a intial guess
    T_teach_repeat_edge = repeat_edge_transforms[repeat_vertex_idx][0][repeat_vertex_time[0]]    

    T_teach_repeat_edge_in_radar = T_radar_robot @ T_teach_repeat_edge @ T_radar_robot.inverse()
    r_repeat_teach_in_teach = T_teach_repeat_edge_in_radar.inverse().r_ba_ina() # inverse?

    # we might need to transform r_repeat_teach to the radar frame
    roll, pitch, yaw = rotation_matrix_to_euler_angles(T_teach_repeat_edge.C_ba().T) # ba means teach_repeat
    r_repeat_teach_in_teach[2] = wrap_angle(yaw)
    print("r_repeat_teach_in_teach:", r_repeat_teach_in_teach.T[0])

    vtr_se2_pose.append(r_repeat_teach_in_teach.T[0])

    intial_guess = torch.from_numpy(np.squeeze(r_repeat_teach_in_teach)).to('cuda')
    # print("intial_guess shape:", intial_guess.shape)

    # gp_state_estimator.setIntialState(intial_guess) # we can comment this out
    state = gp_state_estimator.pairwiseRegistration(teach_frame, repeat_frame)
    direct_se2_pose.append(state)
    norm_state = np.linalg.norm(state[0:2])

    dir_norm.append(norm_state)

    print("direct estimated state:",state)



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
    





# for v, e in PriviledgedIterator(v_start):

#     teach_vertex = v
#     x_teach.append(v.T_v_w.r_ba_ina()[0])
#     y_teach.append(v.T_v_w.r_ba_ina()[1])
#     z_teach.append(v.T_v_w.r_ba_ina()[2])
#     t_teach.append(v.stamp / 1e9)

#     # b_scan_msg = teach_vertex.get_data("radar_b_scan_img")
#     # b_scan_img_ROS = b_scan_msg.b_scan_img

#     # timestamps = np.array(b_scan_msg.timestamps)
#     # azimuth_angles = np.array(b_scan_msg.encoder_values)/16000*2*np.pi

#     # cv_scan_polar = bridge.imgmsg_to_cv2(b_scan_img_ROS)

#     # print("azimuth shape:",azimuth_angles.shape)
#     # print("timestamps shape:",timestamps.shape)

#     # cart_img = radar_polar_to_cartesian(cv_scan_polar, azimuth_angles, radar_resolution, cart_resolution, 640)

#     # plt.imshow(bridge.imgmsg_to_cv2(b_scan_img_ROS),cmap='gray')

#     # plt.imshow(cart_img,cmap='gray')
#     # plt.show()


           
# # I will use the repeat path for now
# v_start = test_graph.get_vertex((repeat, 0))


# previous_time = None


# x_repeat = []
# y_repeat = []
# z_repeat = []
# t_repeat = []

# x_repeat_in_gps = []
# y_repeat_in_gps = []
# z_repeat_in_gps = []

# dist = []
# path_len = 0
# previous_error = 0

# cnt = 0

# for vertex, e in TemporalIterator(v_start): # I am going through all the repeat vertices

#     repeat_vertex = vertex

#     # closest teach vertex
#     teach_vertex = g_utils.get_closest_teach_vertex(repeat_vertex)
#     # print("frame: ", frame)
#     teach_b_scan_msg = teach_vertex.get_data("radar_b_scan_img") # map 
#     teach_b_scan_img_ROS = teach_b_scan_msg.b_scan_img

#     teach_scan_timestamps = np.array(teach_b_scan_msg.timestamps)
#     teach_scan_azimuth_angles = np.array(teach_b_scan_msg.encoder_values)/16000*2*np.pi

#     teach_cv_scan_polar = bridge.imgmsg_to_cv2(teach_b_scan_img_ROS)

#     print("teach scan azimuth shape:",teach_scan_azimuth_angles.shape)
#     print("teach scan timestamps shape:",teach_scan_timestamps.shape)

#     teach_cart_img = radar_polar_to_cartesian(teach_cv_scan_polar, teach_scan_azimuth_angles, radar_resolution, cart_resolution, 640)

#     repeat_b_scan_msg = repeat_vertex.get_data("radar_b_scan_img")
#     repeat_b_scan_img_ROS = repeat_b_scan_msg.b_scan_img

#     repeat_scan_timestamps = np.array(repeat_b_scan_msg.timestamps)
#     repeat_scan_azimuth_angles = np.array(repeat_b_scan_msg.encoder_values)/16000*2*np.pi

#     repeat_cv_scan_polar = bridge.imgmsg_to_cv2(repeat_b_scan_img_ROS)

#     repeat_cart_img = radar_polar_to_cartesian(repeat_cv_scan_polar, repeat_scan_azimuth_angles, radar_resolution, cart_resolution, 640)

#     print("repeat scan azimuth shape:",repeat_scan_azimuth_angles.shape)
#     print("repeat scan timestamps shape:",repeat_scan_timestamps.shape)

#     fig, axs = plt.subplots(1, 3, tight_layout=True)

#     axs[0].imshow(teach_cart_img,cmap='gray')
#     axs[0].set_title('teach')
#     axs[1].imshow(repeat_cart_img,cmap='gray')
#     axs[1].set_title('repeat')


#     # class radarframe:
#     class RadarFrame:
#         def __init__(self, polar, azimuths, timestamps):
#             self.polar = polar[:, :].astype(np.float32) / 255.0
#             self.azimuths=azimuths
#             self.timestamps=timestamps.flatten().astype(np.int64)  


#     teach_frame = RadarFrame(teach_cv_scan_polar, teach_scan_azimuth_angles, teach_scan_timestamps.reshape(-1,1))
#     repeat_frame = RadarFrame(repeat_cv_scan_polar, repeat_scan_azimuth_angles, repeat_scan_timestamps.reshape(-1,1))

#     # state = gp_state_estimator.pairwiseRegistration(teach_frame, repeat_frame)

#     state = one_image_to_one_image(teach_frame, repeat_frame, config_warthog, radar_resolution)
#     norm_state = np.linalg.norm(state[0:2])

#     print("state:",state)
#     print("norm state:",norm_state)

#     # current_pos, current_rot = gp_state_estimator.getAzPosRot()
#     current_pos = state[:2]
#     current_rot = state[2:]

#     # vtr_estimate
#     x_repeat_vtr = repeat_vertex.T_v_w.r_ba_ina()[0]
#     y_repeat_vtr = repeat_vertex.T_v_w.r_ba_ina()[1]
#     z_repeat_vtr = repeat_vertex.T_v_w.r_ba_ina()[2]

#     repeat_se3_pose = repeat_vertex.T_v_w
#     C_v_w = repeat_se3_pose.C_ba()

#     roll_repeat_vtr,pitch_repeat_vtr,yaw_repeat_vtr = rotation_matrix_to_euler_angles(C_v_w)

#     yaw_repeat_vtr = np.array([wrap_angle(yaw_repeat_vtr)])
#     # print("x repeat vtr:",x_repeat_vtr)
#     # print("y repeat vtr:",y_repeat_vtr)
#     # print("yaw repeat vtr:",yaw_repeat_vtr)

#     se2_repeat = np.array([x_repeat_vtr, y_repeat_vtr, yaw_repeat_vtr])

#     print("vtr repeat x y yaw:",x_repeat_vtr,y_repeat_vtr,yaw_repeat_vtr)

#     # print("vtr repeat vertex pose:",repeat_vertex.T_v_w.matrix())

#     x_teach_vtr = teach_vertex.T_v_w.r_ba_ina()[0]
#     y_teach_vtr = teach_vertex.T_v_w.r_ba_ina()[1]
#     z_teach_vtr = teach_vertex.T_v_w.r_ba_ina()[2]

#     teach_se3_pose = teach_vertex.T_v_w
#     C_v_w = teach_se3_pose.C_ba()
#     roll_teach_vtr,pitch_teach_vtr,yaw_teach_vtr = rotation_matrix_to_euler_angles(C_v_w)
#     yaw_teach_vtr = np.array([wrap_angle(yaw_teach_vtr)])

#     se2_teach = np.array([x_teach_vtr, y_teach_vtr, yaw_teach_vtr])

#     difference = se2_repeat - se2_teach

#     norm_difference = np.linalg.norm(difference[0:2])
#     print("vtr difference:",difference.T)
#     print("vtr norm difference:",norm_difference)


#     # print("vtr teach vertex pose:",teach_vertex.T_v_w.matrix())
#     print("vtr teach x y yaw:",x_teach_vtr,y_teach_vtr,yaw_teach_vtr)


#     # print("x_gt teach:",x_gt_teach[0:10])
#     # print("y_gt teach:",y_gt_teach[0:10])

#     # print("x_gt repeat:",x_gt_repeat[0:10])
#     # print("y_gt repeat:",y_gt_repeat[0:10])
#     if cnt == 5:
#         break
#     cnt += 1

    







#     # break





#     # # the things that are available for direct registration
#     # # 1. teach_cart_img and polar plus their timestamps and azimuth angles
#     # # 2. repeat_cart_img and polar plus their timestamps and azimuth angles
#     # # 3. teach_vetrex pose 

#     # teach_pose = teach_vertex.T_v_w

#     # # teach_velocity = teach_vertex.

#     # print("first teach_pose:",teach_pose.matrix())

#     # gp_state_estimator.state_init = teach_pose


#     # # I want to save frame by frame and with repeat timestamps as the frame name
#     # repeat_vertex_time = repeat_vertex.stamp/1e9
    
#     # # repeat point cloud
#     # repeat_new_points, T_v_s = extract_points_from_vertex(repeat_vertex, msg="filtered_point_cloud", return_tf=True)
#     # repeat_new_points = convert_points_to_frame(repeat_new_points, T_v_s.inverse())

#     # #  teach map point cloud
#     # teach_new_points = extract_map_from_vertex(test_graph, repeat_vertex,False)
#     # teach_new_points = convert_points_to_frame(teach_new_points, T_v_s.inverse())
#     # # print(T_v_s.inverse().matrix())


#     # # here is what I want, vertex timestamp + scan to map transform in SE(3) 
#     # transformation = T_v_s.matrix()
#     # # print("transformation to the map:",transformation)

#     # x_repeat.append(vertex.T_v_w.r_ba_ina()[0])
#     # y_repeat.append(vertex.T_v_w.r_ba_ina()[1])
#     # z_repeat.append(vertex.T_v_w.r_ba_ina()[2])
#     # t_repeat.append(repeat_vertex_time)

#     # r_gps = (T_novatel_robot @ vertex.T_v_w).r_ba_ina()
#     # x_repeat_gps = r_gps[0]
#     # y_repeat_gps = r_gps[1]
#     # z_repeat_gps = r_gps[2]

#     # x_repeat_in_gps.append(x_repeat_gps)
#     # y_repeat_in_gps.append(y_repeat_gps)
#     # z_repeat_in_gps.append(z_repeat_gps)
#     # # print("r_gps:",r_gps)


#     # error = signed_distance_to_path(r_gps, path_matrix)
#     # product = error*previous_error
#     # if product<0 and abs(error)>0.05 and abs(previous_error)>0.05:
#     #     error = -1*error

#     # dist.append(error)
#     # previous_error = error
#     # path_len += np.linalg.norm(e.T.r_ba_ina())

#     # break

# # fig, axs = plt.subplots(1, 3, tight_layout=True)
# # axs[0].plot(x_teach, y_teach, label='teach',color='blue')
# # axs[0].set_aspect('equal')  # Ensure equal axis scaling
# # axs[0].plot(x_repeat, y_repeat, label='repeat',color='red')
# # axs[0].set_title('teach vs repeat')
# # axs[0].set_aspect('equal')  # Ensure equal axis scaling
# # # axs[2].imshow(rgb)
# # plt.show()




# # outpath = os.path.join(out_path_folder,f"grassy_teach_radar_association_repeat{repeat}.npz")

# # if SAVE:
# #     # print("I am in the if clause")
# #     paths = [teach_rosbag_path, repeat_rosbag_path]
# #     teach = True
# #     for path in paths:
# #         print("processing path:",path)
# #         fft_data,radar_timestamps,azimuth_angles, azimuth_timestamps_total,cart_imgs = get_radar_scan_images_and_timestamps(path)
        
# #         if teach:
# #             scan_folder = os.path.join(out_path_folder, "teach_radar_scans")
# #             teach = False
# #             print("Teach duration", radar_timestamps[-1] - radar_timestamps[0])
# #         else:
# #             scan_folder = os.path.join(out_path_folder, f"repeat{repeat}_radar_scans")
# #             print(f"Repeat{repeat} duration", radar_timestamps[-1] - radar_timestamps[0])

# #         for timestamp in radar_timestamps:
# #             print("processing scan index:",radar_timestamps.index(timestamp))
# #             # print("radar scan timestamp:",timestamp)
# #             polar_img = fft_data[radar_timestamps.index(timestamp)]

# #             encoder_values = np.array(azimuth_angles[radar_timestamps.index(timestamp)])
# #             # print("encoder_values:",encoder_values)

# #             azimuth_timestamps = np.array(azimuth_timestamps_total[radar_timestamps.index(timestamp)])
# #             # print("azimuth_timestamps:",azimuth_timestamps)

# #             combined = np.vstack((azimuth_timestamps,encoder_values)).T
# #             # print("combined shape:",combined.shape)

# #             if not os.path.exists(scan_folder):
# #                 os.makedirs(scan_folder)
# #                 print(f"Folder '{scan_folder}' created.")

# #             cv2.imwrite(os.path.join(scan_folder,f"{timestamp}.png"), polar_img)
# #             # cv2.imshow("polar_img",cart_imgs[radar_timestamps.index(timestamp)])
# #             np.savetxt(os.path.join(scan_folder,f"{timestamp}.txt"), combined, delimiter=",", fmt='%s')