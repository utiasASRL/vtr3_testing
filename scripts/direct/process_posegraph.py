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

parent_folder = "/home/leonardo/vtr3_testing"

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


config = load_config(os.path.join(parent_folder,'scripts/direct/direct_config.yaml'))

# Access database configuration
db = config['radar_data']['grassy']
db_rosbag_path = db.get('rosbag_path')

# teach_rosbag_path = db_rosbag_path.get('teach')

global repeat
repeat = 1
# repeat_rosbag_path = db_rosbag_path.get(f'repeat{repeat}') # dont think this is needed

# for pose graph
pose_graph_path = db.get('pose_graph_path').get('grassy_t2_r3')
print("pose graph path:",pose_graph_path)

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
out_path_folder = os.path.join(result_folder,f"grassy_t2_r3/") # change path here
if not os.path.exists(out_path_folder):
    os.makedirs(out_path_folder)
    print(f"Folder '{out_path_folder}' created.")
else:
    print(f"Folder '{out_path_folder}' already exists.")    


# if SAVE:
#     paths = [teach_rosbag_path, repeat_rosbag_path]
#     teach = True
#     for path in paths:
#         print("processing path:",path)
#         fft_data,radar_timestamps,azimuth_angles, azimuth_timestamps_total,cart_imgs = get_radar_scan_images_and_timestamps(path)
#         if teach:
#             scan_folder = os.path.join(out_path_folder, "teach_radar_scans")
#             teach = False
#             print("Teach duration", radar_timestamps[-1] - radar_timestamps[0])
#         else:
#             scan_folder = os.path.join(out_path_folder, f"repeat{repeat}_radar_scans")
#             print(f"Repeat{repeat} duration", radar_timestamps[-1] - radar_timestamps[0])

#         for timestamp in radar_timestamps:
#             print("processing scan index:",radar_timestamps.index(timestamp))
#             # print("radar scan timestamp:",timestamp)
#             polar_img = fft_data[radar_timestamps.index(timestamp)]

#             encoder_values = np.array(azimuth_angles[radar_timestamps.index(timestamp)])

#             azimuth_timestamps = np.array(azimuth_timestamps_total[radar_timestamps.index(timestamp)])

#             combined = np.vstack((azimuth_timestamps,encoder_values)).T

#             if not os.path.exists(scan_folder):
#                 os.makedirs(scan_folder)
#                 print(f"Folder '{scan_folder}' created.")

#             cv2.imwrite(os.path.join(scan_folder,f"{timestamp}.png"), polar_img)
#             np.savetxt(os.path.join(scan_folder,f"{timestamp}.txt"), combined, delimiter=",", fmt='%s')

   
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
path_matrix = vtr_path.path_to_matrix(test_graph, PriviledgedIterator(v_start), transform=T_novatel_robot)
print("path matrix shape:",path_matrix.shape)

# x_teach = []
# y_teach = []
# z_teach = []
# t_teach = []

# roll_teach = []
# pitch_teach = []
# yaw_teach = []

# x_repeat_in_gps = []
# y_repeat_in_gps = []
# z_repeat_in_gps = []

# for v, e in PriviledgedIterator(v_start):
#     vertex = v
#     x_teach.append(v.T_v_w.r_ba_ina()[0])
#     y_teach.append(v.T_v_w.r_ba_ina()[1])
#     z_teach.append(v.T_v_w.r_ba_ina()[2])
#     t_teach.append(v.stamp / 1e9)

#     T_gps_w = T_novatel_robot @ vertex.T_v_w
#     r_gps = (T_gps_w).r_ba_ina()
#     x_teach_gps = r_gps[0]
#     y_teach_gps = r_gps[1]
#     z_teach_gps = r_gps[2]

#     C_v_w = v.T_v_w.C_ba()

#     roll, pitch, yaw = rotation_matrix_to_euler_angles(C_v_w)

#     # need to wrap the yaw angle
#     yaw = wrap_to_pi(yaw)

#     roll_teach.append(roll)
#     pitch_teach.append(pitch)
#     yaw_teach.append(yaw)

# BELOW IS THE REPEAT PATH
# I will use the repeat path for now
v_start = test_graph.get_vertex((repeat, 0))

x_teach = []
y_teach = []
z_teach = []
t_teach = []

x_repeat = []
y_repeat = []
z_repeat = []
t_repeat = []

x_repeat_in_gps = []
y_repeat_in_gps = []
z_repeat_in_gps = []

x_repeat_in_gps = []
y_repeat_in_gps = []
z_repeat_in_gps = []

dist = []
path_len = 0
previous_error = 0

# set up the save folder for teach and repeat
teach_folder = os.path.join(out_path_folder, "teach")
if not os.path.exists(teach_folder):
    os.makedirs(teach_folder)
    print(f"Folder '{teach_folder}' created.")
# a couple of things I need to save
# in the teach
# 1. (932,400,1712) images
teach_polar_imgs = []
# 2. (932,400, 1) azimuth angles
teach_azimuth_angles = []
# 3. (932,400, 1) azimuth timestamps
teach_azimuth_timestamps = []
# 4. (932,1) vertex timestamps
teach_vertex_timestamps = []
# 5. Pose at each vertex: (932,4,4)
teach_vertex_transforms = []
# 6. teach vertext time
teach_times = []
# in the repeat
repeat_folder = os.path.join(out_path_folder, f"repeat")
if not os.path.exists(repeat_folder):
    os.makedirs(repeat_folder)
    print(f"Folder '{repeat_folder}' created.")
# 1. (932,400,1712) images
repeat_polar_imgs = []
# 2. (932,400, 1) azimuth angles
repeat_azimuth_angles = []
# 3. (932,400, 1) azimuth timestamps
repeat_azimuth_timestamps = []
# 4. (932,1) vertex timestamps
repeat_vertex_timestamps = []
# 5. relative edge transform at each vertex: (932,4,4)
repeat_edge_transforms = []
# 6. repeat vertex time
repeat_times = []
# needed for image conversion
bridge = CvBridge()

for vertex, e in TemporalIterator(v_start): # I am going through all the repeat vertices
    # print("repeat vertex id:",vertex.id)
    # this vertex is the repeat vertex

    repeat_vertex = vertex
    repeat_vertex_time = repeat_vertex.stamp/1e9
    repeat_times.append(repeat_vertex_time)

    teach_vertex = g_utils.get_closest_teach_vertex(repeat_vertex) 
    print("closest teach vertex id:",teach_vertex.id)
    teach_vertex_time = teach_vertex.stamp/1e9
    teach_times.append(teach_vertex_time)

    teach_b_scan_msg = teach_vertex.get_data("radar_b_scan_img") # map 
    teach_b_scan_img_ROS = teach_b_scan_msg.b_scan_img

    teach_scan_timestamps = np.array(teach_b_scan_msg.timestamps).reshape(-1,1)
    teach_scan_azimuth_angles = (np.array(teach_b_scan_msg.encoder_values)/16000*2*np.pi).reshape(-1,1)

    teach_cv_scan_polar = bridge.imgmsg_to_cv2(teach_b_scan_img_ROS)
    
    teach_polar_imgs.append(teach_cv_scan_polar)
    teach_azimuth_angles.append(teach_scan_azimuth_angles)
    teach_azimuth_timestamps.append(teach_scan_timestamps)
    teach_vertex_timestamps.append(teach_vertex_time)


    T_v_w_teach = teach_vertex.T_v_w
    teach_vertex_transforms.append({teach_vertex_time : T_v_w_teach}) # save it as a dict

    # print("teach vertex transform shape:", np.array(teach_vertex_transforms).shape)
    T_v_w_repeat = repeat_vertex.T_v_w

    T_teach_repeat = T_v_w_teach @ T_v_w_repeat.inverse()

    # print("T_repeat_teach: \n",T_teach_repeat.matrix())

    # now we get to the repeat
    # Alternatively, I can use the edge to get the transformation directly
    repeat_neighbor = repeat_vertex.get_neighbours()
    print("repeat neighbor:",repeat_neighbor)

    for neighbor in repeat_neighbor:
        vertex = neighbor[0]
        edge = neighbor[1]

        if edge.is_spatial(): #and vertex.id == teach_vertex.id:
            print("edge is spatial")
            print("edge from id:",edge.from_id)
            print("edge to id:",edge.to_id) # from repeat to teach which is what I want
           
            T_teach_repeat_edge = edge.T

    repeat_edge_transforms.append({repeat_vertex_time : T_teach_repeat_edge})

            


    print("teach vertex time:", teach_vertex_time) # I will use this as file name for teach image
    print("repeat vertex time:", repeat_vertex_time) # I will use this as file name for repeat image

    repeat_b_scan_msg = repeat_vertex.get_data("radar_b_scan_img")
    repeat_b_scan_img_ROS = repeat_b_scan_msg.b_scan_img

    repeat_scan_timestamps = np.array(repeat_b_scan_msg.timestamps)
    repeat_scan_azimuth_angles = (np.array(repeat_b_scan_msg.encoder_values)/16000*2*np.pi).reshape(-1,1)

    repeat_cv_scan_polar = bridge.imgmsg_to_cv2(repeat_b_scan_img_ROS)

    repeat_polar_imgs.append(repeat_cv_scan_polar)
    repeat_azimuth_angles.append(repeat_scan_azimuth_angles)
    repeat_azimuth_timestamps.append(repeat_scan_timestamps)
    repeat_vertex_timestamps.append(repeat_vertex_time)

    # get T&R estimated path tracking error
    r_gps = (T_novatel_robot @ repeat_vertex.T_v_w).r_ba_ina()
    x_repeat_gps = r_gps[0]
    y_repeat_gps = r_gps[1]
    z_repeat_gps = r_gps[2]

    x_repeat_in_gps.append(x_repeat_gps)
    y_repeat_in_gps.append(y_repeat_gps)
    z_repeat_in_gps.append(z_repeat_gps)
    # print("r_gps:",r_gps)


    error = signed_distance_to_path(r_gps, path_matrix)
    product = error*previous_error
    if product<0 and abs(error)>0.05 and abs(previous_error)>0.05: # TODO this value can be tuned
        error = -1*error

    dist.append(error)
    previous_error = error
    path_len += np.linalg.norm(e.T.r_ba_ina())

# # GPT solution
# dist = correct_sign_flips(dist)
    # break
    

print("path length:",path_len)
# make them into numpy arrays
teach_times = np.array(teach_times).reshape(-1,1)
teach_polar_imgs = np.array(teach_polar_imgs)
teach_azimuth_angles = np.array(teach_azimuth_angles)
teach_azimuth_timestamps = np.array(teach_azimuth_timestamps)
teach_vertex_timestamps = np.array(teach_vertex_timestamps).reshape(-1,1)
teach_vertex_transforms = np.array(teach_vertex_transforms).reshape(-1,1)

repeat_times = np.array(repeat_times).reshape(-1,1)
repeat_polar_imgs = np.array(repeat_polar_imgs)
repeat_azimuth_angles = np.array(repeat_azimuth_angles)
repeat_azimuth_timestamps = np.array(repeat_azimuth_timestamps)
repeat_vertex_timestamps = np.array(repeat_vertex_timestamps).reshape(-1,1)
repeat_edge_transforms = np.array(repeat_edge_transforms).reshape(-1,1)

dist = np.array(dist).reshape(-1,1)

if DEBUG:   
    print("teach times shape:", teach_times.shape)
    print("teach polar imgs shape:", teach_polar_imgs.shape)
    print("teach azimuth angles shape:", teach_azimuth_angles.shape)
    print("teach azimuth timestamps shape:", teach_azimuth_timestamps.shape)
    print("teach vertex timestamps shape:", teach_vertex_timestamps.shape)
    print("teach vertex transforms shape:", teach_vertex_transforms.shape)

    print("repeat times shape:", repeat_times.shape)
    print("repeat polar imgs shape:", repeat_polar_imgs.shape)
    print("repeat azimuth angles shape:", repeat_azimuth_angles.shape)
    print("repeat azimuth timestamps shape:", repeat_azimuth_timestamps.shape)
    print("repeat vertex timestamps shape:", repeat_vertex_timestamps.shape)
    print("repeat edge transforms shape:", repeat_edge_transforms.shape)

    print("estimated path tracking error shape:", dist.shape)

if PLOT:
    plt.figure(0)
    plt.plot(x_repeat, y_repeat, 'r')
    plt.plot(x_repeat_in_gps, y_repeat_in_gps, 'b')
    plt.legend(["Repeat in robot frame", "Repeat in GPS frame"])
    plt.axis('equal')
    plt.grid(True)
    plt.show()



# SAVE TEACH CONTENT IN THE TEACH FOLDER
np.savez_compressed(teach_folder + "/teach.npz",
                    teach_polar_imgs=teach_polar_imgs,
                    teach_azimuth_angles=teach_azimuth_angles,
                    teach_azimuth_timestamps=teach_azimuth_timestamps,
                    teach_vertex_timestamps=teach_vertex_timestamps,
                    teach_vertex_transforms=teach_vertex_transforms,
                    teach_times=teach_times)
# SAVE REPEAT CONTENT IN THE REPEAT FOLDER
np.savez_compressed(repeat_folder + "/repeat.npz",
                    repeat_polar_imgs=repeat_polar_imgs,
                    repeat_azimuth_angles=repeat_azimuth_angles,
                    repeat_azimuth_timestamps=repeat_azimuth_timestamps,
                    repeat_vertex_timestamps=repeat_vertex_timestamps,
                    repeat_edge_transforms=repeat_edge_transforms,
                    repeat_times=repeat_times,
                    dist=dist)
    
    # # repeat point cloud
    # repeat_new_points, T_v_s = extract_points_from_vertex(repeat_vertex, msg="filtered_point_cloud", return_tf=True)
    # repeat_new_points = convert_points_to_frame(repeat_new_points, T_v_s.inverse())

    # #  teach map point cloud
    # teach_new_points = extract_map_from_vertex(test_graph, repeat_vertex,False)
    # teach_new_points = convert_points_to_frame(teach_new_points, T_v_s.inverse())

    # # for calculatng the estimated path tracking error
    # x_repeat.append(repeat_vertex.T_v_w.r_ba_ina()[0])
    # y_repeat.append(vertex.T_v_w.r_ba_ina()[1])
    # z_repeat.append(vertex.T_v_w.r_ba_ina()[2])
    # t_repeat.append(repeat_vertex_time)


    # r_gps = (T_novatel_robot @ vertex.T_v_w).r_ba_ina()
    # x_repeat_gps = r_gps[0]
    # y_repeat_gps = r_gps[1]
    # z_repeat_gps = r_gps[2]

    # x_repeat_in_gps.append(x_repeat_gps)
    # y_repeat_in_gps.append(y_repeat_gps)
    # z_repeat_in_gps.append(z_repeat_gps)
    # # print("r_gps:",r_gps)


    # error = signed_distance_to_path(r_gps, path_matrix)
    # product = error*previous_error
    # if product<0 and abs(error)>0.05 and abs(previous_error)>0.05:
    #     error = -1*error

    # dist.append(error)
    # previous_error = error
    # path_len += np.linalg.norm(e.T.r_ba_ina())


    # break

# print("path length:",path_len)
# plt.figure(0)
# plt.plot(x_repeat, y_repeat, 'r')
# plt.plot(x_repeat_in_gps, y_repeat_in_gps, 'b')
# plt.legend(["Repeat in robot frame", "Repeat in GPS frame"])
# plt.axis('equal')
# plt.grid(True)





    # if SAVE:
    #     pc_timestamp = repeat_vertex.stamp/1e9 # repeat vertex timestamp
    #     if previous_time == None:
    #         previous_time = pc_timestamp

    #     print("vertex timestamp: ", pc_timestamp)
    #     print("repeat point cloud shape:", repeat_new_points.T.shape)
    #     # print("repeat point cloud:", repeat_new_points[:,0:10])
        
    #     # need to find the closest radar image timestamp
    #     radar_timestamps = np.array(radar_timestamps)

    #     print("DEBUG: radar timestamps shape:", radar_timestamps.shape)
    #     radar_idx = np.argmin(np.abs(radar_timestamps - pc_timestamp))
    #     dt = np.abs(radar_timestamps[radar_idx] - pc_timestamp)
    #     print("dt:", dt)
    #     # print("radar idx:", radar_idx)

    #     radar_img = cart_imgs[radar_idx] # this is repeat radar image
    #     radar_img = cv2.cvtColor(radar_img, cv2.COLOR_GRAY2BGR)

    #     radar_img = cv2.flip(radar_img, 0)
    #     radar_img = cv2.flip(radar_img, 1)


    #     print("radar idx: ", radar_idx)
    #     print("radar img shape: ", radar_img.shape)


    #     for point in teach_new_points.T:
    #         # print("point:",point)
    #         x_pt = int(point[0]/cart_resolution) + 256
    #         y_pt = -int(point[1]/cart_resolution) + 256
    #         z_pt = point[2]
    #         # draw point on img
    #         # print("x_pt", x_pt)
    #         # print("y_pt", y_pt)
    #         radius = 1
    #         color = (216,232,57)
    #         thickness = -1       # Filled circle

    #         cv2.circle(radar_img, (y_pt,x_pt), radius, color, thickness)   

    #     # repeat point cloud
    #     for point in repeat_new_points.T:
    #         x_pt = int(point[0]/cart_resolution) + 256
    #         y_pt = -int(point[1]/cart_resolution) + 256
    #         z_pt = point[2]
    #         # draw point on img
    #         # print("x_pt", x_pt)
    #         # print("y_pt", y_pt)
    #         radius = 1
    #         color = (0,0,255)
    #         thickness = -1       # Filled circle

    #         cv2.circle(radar_img, (y_pt,x_pt), radius, color, thickness)

    #     # just before the write frame, lets flip it again lol
    #     radar_img = cv2.flip(radar_img, -1)

    #     # how many frames in sec
    #     time_elapsed = pc_timestamp - previous_time    
    #     print("time elapsed between last vertex to current one: ", time_elapsed)

    #     frame_name = str(repeat_vertex_time) + ".png"

    #     frame_path = os.path.join(out_path_folder,"frames/")
    #     if not os.path.exists(frame_path):
    #         os.makedirs(frame_path)
    #         print(f"Folder '{frame_path}' created.")
    #     out_frame_name = os.path.join(frame_path,frame_name)
    #     print("writing frame to this location:", out_frame_name)
    #     cv2.imwrite(out_frame_name, radar_img)

        
    #     # if time_elapsed == 0:
    #     #     n_frames = 1
    #     # else:
    #     #     n_frames = int(time_elapsed * frame_rate)

    #     # # n_frames = 1
    #     # print("I need to write ", n_frames, "frames")
    #     # for i in range(n_frames):
    #     #     video_writer.write(radar_img)
    #     #     frame += 1

    #     previous_time = pc_timestamp

        

# print(f"Path {repeat} was {path_len:.3f}m long")


if PLOT:
    fontsize = 20
    plt.figure(1)
    rmse = np.sqrt(np.trapz(np.array(dist)**2, t_repeat) / (t_repeat[-1] - t_repeat[0]))

    # rmse = np.sqrt(np.mean(np.array(dist)**2))
    max = np.max(np.abs(dist))

    # print("dist:",dist)
    # reset t_repeat to 0
    t_repeat = [t - t_repeat[0] for t in t_repeat]
    print("t_repeat last:",t_repeat[-1])
    plt.plot(t_repeat, dist, label=f"Estimated RMSE: {rmse:.3f}m for Repeat {repeat} Max Error: {max:.3f}m")
    # # load the ppk data
    ppk_data = np.load(f"{out_path_folder}repeat_path_tracking_error.npz")
    t_repeat_ppk = ppk_data['t_repeat_ppk']
    distances_teach_repeat_ppk = ppk_data['distances_teach_repeat_ppk']
    plt.plot(t_repeat_ppk, distances_teach_repeat_ppk, label=f"PPK RMSE: {np.sqrt(np.mean(distances_teach_repeat_ppk**2)):.3f}m for Repeat {repeat} Max Error: {np.max(np.abs(distances_teach_repeat_ppk)):.3f}m")
    


#     plt.legend(fontsize=fontsize)
#     plt.ylabel("RTR Estimated Path Tracking Error (m)",fontsize=fontsize)
#     plt.xlabel("Time (s)",fontsize=fontsize)
#     plt.title("RTR Estimated Path Tracking Error",fontsize=fontsize)
#     plt.ylim(-0.6,0.6)
#     # plt.axis('equal')
#     plt.grid(True)

#     plt.show()







