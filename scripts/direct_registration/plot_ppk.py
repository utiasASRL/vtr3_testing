import rosbags
# import cv2
# from cv_bridge import CvBridge
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from matplotlib import pyplot as plt

import utm

# print current working directory
import os
print("Current working dir", os.getcwd())

import sys
# Insert path at index 0 so it's searched first
parent_folder = "/home/samqiao/ASRL/vtr3_testing"
sys.path.insert(0, parent_folder)
# sys.path.insert(0, "scripts")
# sys.path.insert(0, "deps")

# from radar.utils.helper import get_xyt_gps
from deps.path_tracking_error.fcns import *

from scipy import interpolate
import csv

import yaml

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


teach_ppk_path = "localization_data/ppk/grassy_t2/grassy_t2.txt"

teach_ppk_path = "localization_data/ppk/mars_t1/mars_t1.txt"

teach_raw_path = "localization_data/ppk/grassy_t2/grassy2_0_BESTPOS.ASCII"

repeat_ppk_path = "localization_data/ppk/grassy_t3/grassy_t3.txt"
repeat_raw_path = "localization_data/ppk/grassy_t3/grassy3_0_BESTPOS.ASCII"

x_ppk_teach, y_ppk_teach = read_PPK_file(teach_ppk_path)

# I want to get the path length
path_length = get_path_distance_from_gps(x_ppk_teach,y_ppk_teach)
print("Path length:",path_length)

# just want to plot x,y
plt.scatter(x_ppk_teach, y_ppk_teach, label='teach ppk f grassy 2',marker='*',s=3)
plt.axis('equal')
plt.legend(fontsize=20)
plt.xlabel('x (m)',fontsize=20)
plt.ylabel('y (m)',fontsize=20)
plt.xticks(fontsize=20-5)
plt.yticks(fontsize=20-5)
plt.title('PPK comparison',fontsize=20)

plt.show()


# x_raw_teach, y_raw_teach = read_ASCII_file(teach_raw_path)

# x_ppk_repeat, y_ppk_repeat = read_PPK_file(repeat_ppk_path)
# x_raw_repeat, y_raw_repeat = read_ASCII_file(repeat_raw_path)


# # I want to calculate the RMSE and Max error of the repeat path to the teach path
# # lets use raw for testing
# # x_ppk_teach = x_raw_teach
# # y_ppk_teach = y_raw_teach

# # x_ppk_repeat = x_raw_repeat
# # y_ppk_repeat = y_raw_repeat

# z_ppk_teach = np.zeros_like(x_ppk_teach)

# pose_teach = np.vstack((x_ppk_teach,y_ppk_teach,z_ppk_teach)).T
# print("pose_teach shape:",pose_teach.shape)

# points_teach = path_to_matrix(pose_teach)
# print("points_teach shape:",points_teach.shape)

# z_ppk_repeat = np.zeros_like(x_ppk_repeat)
# pose_repeat = np.vstack((x_ppk_repeat,y_ppk_repeat,z_ppk_repeat)).T
# print("pose_repeat shape:",pose_repeat.shape)

# # distances_teach_repeat = distances_to_path(pose_repeat,points_teach)
# # hey I need to do a check here so that the error does not jump around (I also want to keep the time steps)
# previous_error = 0
# distances_teach_repeat = []
# for i in range(len(pose_repeat)):
#     # print("i:",i)
#     error = signed_distance_to_path(pose_repeat[i],points_teach)
#     product = error*previous_error
#     if product<0 and abs(error)>0.05 and abs(previous_error)>0.05:
#         error = -1*error
#     distances_teach_repeat.append(error)
#     previous_error = error

# distances_teach_repeat = np.array(distances_teach_repeat)
# # # filter out the outliers
# # t_repeat = t_repeat[distances_teach_repeat < 1]
# # distances_teach_repeat = distances_teach_repeat[distances_teach_repeat < 1]


# Max = np.max(np.abs(distances_teach_repeat))
# RMSE = np.sqrt(np.mean(distances_teach_repeat**2))
# print(f"RMSE: {RMSE} m, Max: {Max} m")
    
#     # fontsize = 20
#     # plt.figure(figsize=(10,7))
#     # plt.title("Repeat Individual Path-tracking error over time",fontsize=fontsize)
#     # plt.scatter(t_repeat,distances_teach_repeat)
#     # plt.xlabel('Time [s]')
#     # # plt.axis('equal')
#     # plt.legend([f"Path tracking error with mean {round(np.mean(distances_teach_repeat),3)} m, max {round(Max,3)} m"],fontsize=fontsize-5)
#     # plt.grid()
#     # plt.ylabel('Distance to TEACH path [m]')

#     # plt.show()



# if PLOT:    
#     fontsize = 20
#     # ax = plt.gca()
#     # ax.set_aspect('equal', adjustable='box')
#     plt.figure(0)
#     # # teach path
#     plt.scatter(x_ppk_teach, y_ppk_teach, label='teach ppk f grassy 2',marker='*',s=3)
#     # # plt.scatter(x_simulated_teach,y_simulated_teach, label='teach simulated rtk f',marker='*',s=3)
#     plt.scatter(x_ppk_repeat,y_ppk_repeat, label='repeat ppk f grassy 3',marker='*',s=3)

#     # plt.scatter(x_raw_teach, y_raw_teach, label='teach raw rtk f',marker='*',s=3)
#     # plt.scatter(x_raw_repeat, y_raw_repeat, label='repeat raw rtk f',marker='*',s=3)



#     plt.axis('equal')
#     plt.legend(fontsize=fontsize)
#     plt.xlabel('x (m)',fontsize=fontsize)
#     plt.ylabel('y (m)',fontsize=fontsize)
#     plt.xticks(fontsize=fontsize-5)
#     plt.yticks(fontsize=fontsize-5)
#     plt.title('PPK comparison',fontsize=fontsize)
#     plt.grid(True)

#     plt.figure(1)
#     # I want to plot the path tracking error
#     t_repeat = np.arange(len(distances_teach_repeat))/10
#     print("t_repeat last element:",t_repeat[-1])
#     plt.plot(t_repeat, distances_teach_repeat, label=f"RMSE: {RMSE:.3f}m for Repeat {repeat} Max Error: {Max:.3f}m")

#     np.savez(f"{out_path_folder}repeat{repeat}_path_tracking_error.npz",t_repeat_ppk=t_repeat,distances_teach_repeat_ppk=distances_teach_repeat)
#     plt.grid()

#     plt.xlabel('Time (s)',fontsize=fontsize)
#     plt.ylabel('RTR GPS Path Tracking Error (m)',fontsize=fontsize)
#     plt.title(f"Repeat {repeat}: GPS Path-tracking Error over time",fontsize=fontsize)
#     plt.ylim(-0.6,0.6)
#     plt.xticks(fontsize=fontsize-5)
#     plt.legend(fontsize=fontsize)

#     plt.show()

