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
from vtr_utils.plot_utils import convert_points_to_frame, extract_map_from_vertex, downsample, extract_points_from_vertex, range_crop
import time

import yaml

from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images


print("Current working dir", os.getcwd())

# initlize the video writer
# Parameters for the video writer
frame_rate = 60.0  # Frames per second
frame_size = (512, 512)  # Frame size (width, height) of the video
codec = cv2.VideoWriter_fourcc(*'XVID')  # Video codec (XVID, MJPG, etc.)

parent_folder = "/home/samqiao/ASRL/vtr3_testing"

# Insert path at index 0 so it's searched first
sys.path.insert(0, parent_folder)


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

config = load_config(os.path.join(parent_folder,'scripts/direct/direct_configs/direct_config_sam.yaml'))

# Access database configuration
db = config['radar_data']['mars']
# db_rosbag_path = db.get('grassy_rosbag_path')

# teach_rosbag_path = db_rosbag_path.get('teach')

# global repeat
# repeat = 1
# repeat_rosbag_path = db_rosbag_path.get(f'repeat{repeat}') # dont think this is needed

sequence = "mars_t1_r2"

# for pose graph
pose_graph_path = db.get('pose_graph_path').get(sequence)
print("pose graph path:",pose_graph_path)

db_bool = config['bool']
SAVE = db_bool.get('SAVE')
print("SAVE:",SAVE)
PLOT = db_bool.get('PLOT')

result_folder = config.get('output')

# change here
out_path_folder = os.path.join(result_folder,sequence)
if not os.path.exists(out_path_folder):
    os.makedirs(out_path_folder)
    print(f"Folder '{out_path_folder}' created.")
else:
    print(f"Folder '{out_path_folder}' already exists.")    



factory = Rosbag2GraphFactory(pose_graph_path)

test_graph = factory.buildGraph()
print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

g_utils.set_world_frame(test_graph, test_graph.root)


# I will use the repeat path for now
global repeat 
repeat = 1
v_start = test_graph.get_vertex((repeat, 0))


# I want to sepcify the timeframe where I want the frames to be generated
starting_time = 1736886299.265562 # those will be in ros nano seconds
ending_time = 1736886302.5128138

# the same image will have the 

bridge = CvBridge()

for vertex, e in TemporalIterator(v_start):
    # this vertex is the repeat vertex

    repeat_vertex = vertex

    teach_vertex = g_utils.get_closest_teach_vertex(repeat_vertex)

    # I want to save frame by frame and with repeat timestamps as the frame name
    repeat_vertex_time = repeat_vertex.stamp/1e9

    print(f"-------------------Processing repeat vertex with time {repeat_vertex_time} ---------------------- in seconds")

    if repeat_vertex_time < starting_time or repeat_vertex_time > ending_time:
        print(f"Skipping vertex at time {repeat_vertex_time} as it is outside the specified range.")
        continue
    
    # repeat point cloud
    repeat_new_points, T_v_s = extract_points_from_vertex(repeat_vertex, msg="filtered_point_cloud", return_tf=True)
    repeat_new_points= convert_points_to_frame(repeat_new_points, T_v_s.inverse())

    #  teach map point cloud
    teach_new_points, _= extract_map_from_vertex(test_graph, repeat_vertex,False)
    teach_new_points = convert_points_to_frame(teach_new_points, T_v_s.inverse())
    # print(T_v_s.inverse().matrix())


    print("teach_new_points shape:", teach_new_points.shape)
    # teach_new_points = convert_points_to_frame(teach_new_points, T_v_s.inverse())
    # print("teach new points:",teach_new_points.T[0])
    # print("teach new points: X",teach_new_points[0][0])
    # print("teach new points: Y",teach_new_points[0][1])

    print("repeat point cloud shape:", repeat_new_points.T.shape)
    # print("repeat point cloud:", repeat_new_points[:,0:10])
    
    # we can get the radar image from the posegraph as well directly
    repeat_b_scan_msg = repeat_vertex.get_data("radar_b_scan_img")
    repeat_b_scan_img_ROS = repeat_b_scan_msg.b_scan_img

    repeat_scan_timestamps = np.array(repeat_b_scan_msg.timestamps)
    repeat_scan_azimuth_angles = (np.array(repeat_b_scan_msg.encoder_values)/16000*2*np.pi).reshape(-1,1)

    repeat_cv_scan_polar = bridge.imgmsg_to_cv2(repeat_b_scan_img_ROS)

    import pyboreas as pb

    class RadarFrame:
        def __init__(self, polar, azimuths, timestamps):
            self.polar = polar[:, :].astype(np.float32) / 255.0
            self.azimuths=azimuths
            self.timestamps=timestamps.flatten().astype(np.int64) 

    repeat_frame = RadarFrame(repeat_cv_scan_polar,repeat_scan_azimuth_angles, repeat_scan_timestamps)

    radar_img = pb.utils.radar.radar_polar_to_cartesian(repeat_frame.azimuths.astype(np.float32), repeat_frame.polar, 0.040308, 0.2384, 512, False, True)

    # convert radar image to BGR
    radar_img = cv2.cvtColor(radar_img, cv2.COLOR_GRAY2RGB)

    radar_img = cv2.flip(radar_img, 0)
    radar_img = cv2.flip(radar_img, 1)


    cart_resolution = 0.2384

    # print("teach_new_points shape:", teach_new_points.shape)
    # print("teach point", teach_new_points)
    # print("repeat_new_points shape:", repeat_new_points.shape)
    # print("repeat point", repeat_new_points)


    # teach point cloud
    for point in teach_new_points.T:
        # print("point:",point)
        x_pt = int(point[0]/cart_resolution) + 256
        y_pt = -int(point[1]/cart_resolution) + 256
        z_pt = point[2]
        # draw point on img
        # print("x_pt", x_pt)
        # print("y_pt", y_pt)
        radius = 1
        color = (0,128,128)
        thickness = -1       # Filled circle

        cv2.circle(radar_img, (y_pt,x_pt), radius, color, thickness)   

    # repeat point cloud
    for point in repeat_new_points.T:
        x_pt = int(point[0]/cart_resolution) + 256
        y_pt = -int(point[1]/cart_resolution) + 256
        z_pt = point[2]
        # draw point on img
        # print("x_pt", x_pt)
        # print("y_pt", y_pt)
        radius = 1
        color = (255,0,0)
        thickness = -1       # Filled circle

        cv2.circle(radar_img, (y_pt,x_pt), radius, color, thickness)

    # just before the write frame, lets flip it again lol
    radar_img = cv2.flip(radar_img, -1)



    frame_name = str(repeat_vertex_time) + ".png"

    frame_path = os.path.join(out_path_folder,"error_regions/region7/")
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
        print(f"Folder '{frame_path}' created.")
    out_frame_name = os.path.join(frame_path,frame_name)
    print("writing frame to this location:", out_frame_name)


    # before writing the frame, lets see it
    plt.imshow(radar_img)
    # plt.title(f"Radar Image at {repeat_vertex_time} seconds")
    plt.axis('off')  # Hide the axes
    # plt.show()

    # radar_img_uint8 = (radar_img * 255).astype(np.uint8) 

    # cv2.imwrite(out_frame_name, radar_img)

    plt.savefig(os.path.join(out_path_folder,frame_path,f"{repeat_vertex_time}.png"),bbox_inches='tight',pad_inches=0)

    plt.clf()  # Clear the current figure to avoid overlap in the next iteration

    # break

    
    # if time_elapsed == 0:
    #     n_frames = 1
    # else:
    #     n_frames = int(time_elapsed * frame_rate)

    # # n_frames = 1
    # print("I need to write ", n_frames, "frames")
    # for i in range(n_frames):
    #     video_writer.write(radar_img)
    #     frame += 1

    # previous_time = pc_timestamp

    # break






