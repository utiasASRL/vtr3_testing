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


print("Current working dir", os.getcwd())

# initlize the video writer
# Parameters for the video writer
frame_rate = 60.0  # Frames per second
frame_size = (512, 512)  # Frame size (width, height) of the video
codec = cv2.VideoWriter_fourcc(*'XVID')  # Video codec (XVID, MJPG, etc.)

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

# outpath = os.path.join(out_path_folder,f"grassy_teach_radar_association_repeat{repeat}.npz")

if SAVE:
    # print("I am in the if clause")
    paths = [teach_rosbag_path, repeat_rosbag_path]
    teach = True
    for path in paths:
        print("processing path:",path)
        fft_data,radar_timestamps,azimuth_angles, azimuth_timestamps_total,cart_imgs = get_radar_scan_images_and_timestamps(path)
        
        if teach:
            scan_folder = os.path.join(out_path_folder, "teach_radar_scans")
            teach = False
            print("Teach duration", radar_timestamps[-1] - radar_timestamps[0])
        else:
            scan_folder = os.path.join(out_path_folder, f"repeat{repeat}_radar_scans")
            print(f"Repeat{repeat} duration", radar_timestamps[-1] - radar_timestamps[0])

        for timestamp in radar_timestamps:
            print("processing scan index:",radar_timestamps.index(timestamp))
            # print("radar scan timestamp:",timestamp)
            polar_img = fft_data[radar_timestamps.index(timestamp)]

            encoder_values = np.array(azimuth_angles[radar_timestamps.index(timestamp)])
            # print("encoder_values:",encoder_values)

            azimuth_timestamps = np.array(azimuth_timestamps_total[radar_timestamps.index(timestamp)])
            # print("azimuth_timestamps:",azimuth_timestamps)

            combined = np.vstack((azimuth_timestamps,encoder_values)).T
            # print("combined shape:",combined.shape)

            if not os.path.exists(scan_folder):
                os.makedirs(scan_folder)
                print(f"Folder '{scan_folder}' created.")

            cv2.imwrite(os.path.join(scan_folder,f"{timestamp}.png"), polar_img)
            # cv2.imshow("polar_img",cart_imgs[radar_timestamps.index(timestamp)])
            np.savetxt(os.path.join(scan_folder,f"{timestamp}.txt"), combined, delimiter=",", fmt='%s')

    # np.savez_compressed(outpath,radar_imgs = radar_imgs, radar_times = radar_times)
# else:
#     # print("I am in the else clause",SAVE)
#     data = np.load(outpath,allow_pickle=True)
#     radar_imgs = data['radar_imgs']
#     radar_times = data['radar_times']

#     print("radar_imgs shape:", radar_imgs.shape)
#     print("radar_times shape:", radar_times.shape)

#     print("first radar time:",radar_times[0])
#     print("last radar time:",radar_times[-1])
#     radar_duration = radar_times[-1] - radar_times[0]
#     print("radar duration:",radar_duration)


# first_radar_time = radar_times[0]
# first_radar_img = radar_imgs[0]

# plt.imshow(first_radar_img,cmap='gray', vmin=0, vmax=255)
# print(first_radar_img.shape)
# plt.title("First Radar Image")
# plt.axis('off')

# plt.show()
 

print("-------- begin pose graph processing --------")
# Pose graph processing
factory = Rosbag2GraphFactory(pose_graph_path)

test_graph = factory.buildGraph()
# plot_graph(test_graph)
# plt.show()
print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

g_utils.set_world_frame(test_graph, test_graph.root)


# I will use the repeat path for now
v_start = test_graph.get_vertex((repeat, 0))

path_matrix = vtr_path.path_to_matrix(test_graph, PriviledgedIterator(v_start))

# # for i in range(test_graph.major_id + 1):
# #     v_start = test_graph.get_vertex((i, 0))
# #     paused = True
# frame = 0

# # if it was 50 frames per sec
# # we need to have a frame every 0.02 seconds 200 ms 
# # we need to calculate how many frames need to be written within the dt and write them

previous_time = None

for vertex, e in TemporalIterator(v_start):
    print("repeat vertex id:",vertex.id)
    # this vertex is the repeat vertex

    repeat_vertex = vertex

    teach_vertex = g_utils.get_closest_teach_vertex(repeat_vertex)
    # print("frame: ", frame)

    # I want to save frame by frame and with repeat timestamps as the frame name
    repeat_vertex_time = repeat_vertex.stamp/1e9
    
    # repeat point cloud
    repeat_new_points, T_v_s = extract_points_from_vertex(repeat_vertex, msg="filtered_point_cloud", return_tf=True)
    repeat_new_points = convert_points_to_frame(repeat_new_points, T_v_s.inverse())

    #  teach map point cloud
    teach_new_points = extract_map_from_vertex(test_graph, repeat_vertex,False)
    teach_new_points = convert_points_to_frame(teach_new_points, T_v_s.inverse())
    # print(T_v_s.inverse().matrix())


    # print("teach_new_points shape:", teach_new_points.shape)
    # teach_new_points = convert_points_to_frame(teach_new_points, T_v_s.inverse())

    # here is what I want, vertex timestamp + scan to map transform in SE(3) 



    if SAVE:
        pc_timestamp = repeat_vertex.stamp/1e9 # repeat vertex timestamp
        if previous_time == None:
            previous_time = pc_timestamp

        print("vertex timestamp: ", pc_timestamp)
        print("repeat point cloud shape:", repeat_new_points.T.shape)
        # print("repeat point cloud:", repeat_new_points[:,0:10])
        
        # need to find the closest radar image timestamp
        radar_timestamps = np.array(radar_timestamps)

        print("DEBUG: radar timestamps shape:", radar_timestamps.shape)
        radar_idx = np.argmin(np.abs(radar_timestamps - pc_timestamp))
        dt = np.abs(radar_timestamps[radar_idx] - pc_timestamp)
        print("dt:", dt)
        # print("radar idx:", radar_idx)

        radar_img = cart_imgs[radar_idx] # this is repeat radar image
        radar_img = cv2.cvtColor(radar_img, cv2.COLOR_GRAY2BGR)

        radar_img = cv2.flip(radar_img, 0)
        radar_img = cv2.flip(radar_img, 1)


        print("radar idx: ", radar_idx)
        print("radar img shape: ", radar_img.shape)

        cart_resolution = 0.2384

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
            color = (216,232,57)
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
            color = (0,0,255)
            thickness = -1       # Filled circle

            cv2.circle(radar_img, (y_pt,x_pt), radius, color, thickness)

        # just before the write frame, lets flip it again lol
        radar_img = cv2.flip(radar_img, -1)

        # how many frames in sec
        time_elapsed = pc_timestamp - previous_time    
        print("time elapsed between last vertex to current one: ", time_elapsed)

        frame_name = str(repeat_vertex_time) + ".png"

        frame_path = os.path.join(out_path_folder,"frames/")
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
            print(f"Folder '{frame_path}' created.")
        out_frame_name = os.path.join(frame_path,frame_name)
        print("writing frame to this location:", out_frame_name)
        cv2.imwrite(out_frame_name, radar_img)

        
        # if time_elapsed == 0:
        #     n_frames = 1
        # else:
        #     n_frames = int(time_elapsed * frame_rate)

        # # n_frames = 1
        # print("I need to write ", n_frames, "frames")
        # for i in range(n_frames):
        #     video_writer.write(radar_img)
        #     frame += 1

        previous_time = pc_timestamp

        # break






