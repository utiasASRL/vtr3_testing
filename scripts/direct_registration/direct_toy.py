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
import gp_doppler as gpd

from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images


print("Current working dir", os.getcwd())

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

# xy plots of baselink and transformed points


# # test let T be identity
# T_robot_novatel = np.eye(4)


def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file.

    :param config_path: Path to the YAML configuration file.
    :return: A dictionary representing the configuration.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

# print current working directory
print("Current working dir", Path.cwd())

config = load_config(os.path.join(parent_folder,'scripts/direct_registration/direct_config.yaml'))

# Access database configuration
db = config['radar_data']['grassy']
db_rosbag_path = db.get('rosbag_path')

teach_rosbag_path = db_rosbag_path.get('teach')

global repeat
repeat = 1
repeat_rosbag_path = db_rosbag_path.get(f'repeat{repeat}') # dont think this is needed

# for pose graph
pose_graph_path = db.get('pose_graph_path').get('grassy_t2_r3')
print("pose graph path:",pose_graph_path)

db_bool = config['bool']
SAVE = db_bool.get('SAVE')
SAVE = False
print("SAVE:",SAVE)
PLOT = db_bool.get('PLOT')

result_folder = config.get('output')

# change here
out_path_folder = os.path.join(result_folder,f"grassy_t2_r{repeat}/")
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
config_warthog = load_config(os.path.join(parent_folder,'scripts/direct_registration/warthog_config.yaml'))


# intialize the GP state estimator
gp_state_estimator = gpd.GPStateEstimator(config_warthog, radar_resolution)


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

x_teach = []
y_teach = []
z_teach = []
t_teach = []

bridge = CvBridge()

for v, e in PriviledgedIterator(v_start):

    teach_vertex = v
    x_teach.append(v.T_v_w.r_ba_ina()[0])
    y_teach.append(v.T_v_w.r_ba_ina()[1])
    z_teach.append(v.T_v_w.r_ba_ina()[2])
    t_teach.append(v.stamp / 1e9)

    # b_scan_msg = teach_vertex.get_data("radar_b_scan_img")
    # b_scan_img_ROS = b_scan_msg.b_scan_img

    # timestamps = np.array(b_scan_msg.timestamps)
    # azimuth_angles = np.array(b_scan_msg.encoder_values)/16000*2*np.pi

    # cv_scan_polar = bridge.imgmsg_to_cv2(b_scan_img_ROS)

    # print("azimuth shape:",azimuth_angles.shape)
    # print("timestamps shape:",timestamps.shape)

    # cart_img = radar_polar_to_cartesian(cv_scan_polar, azimuth_angles, radar_resolution, cart_resolution, 640)

    # plt.imshow(bridge.imgmsg_to_cv2(b_scan_img_ROS),cmap='gray')

    # plt.imshow(cart_img,cmap='gray')
    # plt.show()


           
# I will use the repeat path for now
v_start = test_graph.get_vertex((repeat, 0))

# # for i in range(test_graph.major_id + 1):
# #     v_start = test_graph.get_vertex((i, 0))
# #     paused = True
# frame = 0

# # if it was 50 frames per sec
# # we need to have a frame every 0.02 seconds 200 ms 
# # we need to calculate how many frames need to be written within the dt and write them

previous_time = None


x_repeat = []
y_repeat = []
z_repeat = []
t_repeat = []

x_repeat_in_gps = []
y_repeat_in_gps = []
z_repeat_in_gps = []

dist = []
path_len = 0
previous_error = 0

for vertex, e in TemporalIterator(v_start): # I am going through all the repeat vertices
    # print("repeat vertex id:",vertex.id)
    # this vertex is the repeat vertex

    repeat_vertex = vertex

    # closest teach vertex
    teach_vertex = g_utils.get_closest_teach_vertex(repeat_vertex)
    # print("frame: ", frame)
    teach_b_scan_msg = teach_vertex.get_data("radar_b_scan_img")
    teach_b_scan_img_ROS = teach_b_scan_msg.b_scan_img

    teach_scan_timestamps = np.array(teach_b_scan_msg.timestamps)
    teach_scan_azimuth_angles = np.array(teach_b_scan_msg.encoder_values)/16000*2*np.pi

    teach_cv_scan_polar = bridge.imgmsg_to_cv2(teach_b_scan_img_ROS)

    print("teach scan azimuth shape:",teach_scan_azimuth_angles.shape)
    print("teach scan timestamps shape:",teach_scan_timestamps.shape)

    teach_cart_img = radar_polar_to_cartesian(teach_cv_scan_polar, teach_scan_azimuth_angles, radar_resolution, cart_resolution, 640)

    repeat_b_scan_msg = repeat_vertex.get_data("radar_b_scan_img")
    repeat_b_scan_img_ROS = repeat_b_scan_msg.b_scan_img

    repeat_scan_timestamps = np.array(repeat_b_scan_msg.timestamps)
    repeat_scan_azimuth_angles = np.array(repeat_b_scan_msg.encoder_values)/16000*2*np.pi

    repeat_cv_scan_polar = bridge.imgmsg_to_cv2(repeat_b_scan_img_ROS)

    repeat_cart_img = radar_polar_to_cartesian(repeat_cv_scan_polar, repeat_scan_azimuth_angles, radar_resolution, cart_resolution, 640)

    print("repeat scan azimuth shape:",repeat_scan_azimuth_angles.shape)
    print("repeat scan timestamps shape:",repeat_scan_timestamps.shape)

    # fig, axs = plt.subplots(1, 3, tight_layout=True)

    # axs[0].imshow(teach_cart_img,cmap='gray')
    # axs[0].set_title('teach')
    # axs[1].imshow(repeat_cart_img,cmap='gray')
    # axs[1].set_title('repeat')


    # the things that are available for direct registration
    # 1. teach_cart_img and polar plus their timestamps and azimuth angles
    # 2. repeat_cart_img and polar plus their timestamps and azimuth angles
    # 3. teach_vetrex pose 

    teach_pose = teach_vertex.T_v_w

    # teach_velocity = teach_vertex.

    print("first teach_pose:",teach_pose.matrix())

    gp_state_estimator.state_init = teach_pose


    # I want to save frame by frame and with repeat timestamps as the frame name
    repeat_vertex_time = repeat_vertex.stamp/1e9
    
    # repeat point cloud
    repeat_new_points, T_v_s = extract_points_from_vertex(repeat_vertex, msg="filtered_point_cloud", return_tf=True)
    repeat_new_points = convert_points_to_frame(repeat_new_points, T_v_s.inverse())

    #  teach map point cloud
    teach_new_points = extract_map_from_vertex(test_graph, repeat_vertex,False)
    teach_new_points = convert_points_to_frame(teach_new_points, T_v_s.inverse())
    # print(T_v_s.inverse().matrix())


    # here is what I want, vertex timestamp + scan to map transform in SE(3) 
    transformation = T_v_s.matrix()
    # print("transformation to the map:",transformation)

    x_repeat.append(vertex.T_v_w.r_ba_ina()[0])
    y_repeat.append(vertex.T_v_w.r_ba_ina()[1])
    z_repeat.append(vertex.T_v_w.r_ba_ina()[2])
    t_repeat.append(repeat_vertex_time)

    r_gps = (T_novatel_robot @ vertex.T_v_w).r_ba_ina()
    x_repeat_gps = r_gps[0]
    y_repeat_gps = r_gps[1]
    z_repeat_gps = r_gps[2]

    x_repeat_in_gps.append(x_repeat_gps)
    y_repeat_in_gps.append(y_repeat_gps)
    z_repeat_in_gps.append(z_repeat_gps)
    # print("r_gps:",r_gps)


    error = signed_distance_to_path(r_gps, path_matrix)
    product = error*previous_error
    if product<0 and abs(error)>0.05 and abs(previous_error)>0.05:
        error = -1*error

    dist.append(error)
    previous_error = error
    path_len += np.linalg.norm(e.T.r_ba_ina())

    break

# fig, axs = plt.subplots(1, 3, tight_layout=True)
# axs[0].plot(x_teach, y_teach, label='teach',color='blue')
# axs[0].set_aspect('equal')  # Ensure equal axis scaling
# axs[0].plot(x_repeat, y_repeat, label='repeat',color='red')
# axs[0].set_title('teach vs repeat')
# axs[0].set_aspect('equal')  # Ensure equal axis scaling
# # axs[2].imshow(rgb)
# plt.show()




# outpath = os.path.join(out_path_folder,f"grassy_teach_radar_association_repeat{repeat}.npz")

# if SAVE:
#     # print("I am in the if clause")
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
#             # print("encoder_values:",encoder_values)

#             azimuth_timestamps = np.array(azimuth_timestamps_total[radar_timestamps.index(timestamp)])
#             # print("azimuth_timestamps:",azimuth_timestamps)

#             combined = np.vstack((azimuth_timestamps,encoder_values)).T
#             # print("combined shape:",combined.shape)

#             if not os.path.exists(scan_folder):
#                 os.makedirs(scan_folder)
#                 print(f"Folder '{scan_folder}' created.")

#             cv2.imwrite(os.path.join(scan_folder,f"{timestamp}.png"), polar_img)
#             # cv2.imshow("polar_img",cart_imgs[radar_timestamps.index(timestamp)])
#             np.savetxt(os.path.join(scan_folder,f"{timestamp}.txt"), combined, delimiter=",", fmt='%s')