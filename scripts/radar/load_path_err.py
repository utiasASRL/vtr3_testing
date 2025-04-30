import rosbags
import cv2
from cv_bridge import CvBridge
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from matplotlib import pyplot as plt

import utm
import numpy as np

import sys
# Insert path at index 0 so it's searched first
sys.path.insert(0, "scripts")

from radar.utils.helper import *

import yaml
import os
import re 


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

config = load_config(os.path.join(parent_folder,'scripts/config.yaml'))

# Access database configuration
db = config['radar_data']
db_loop = db.get('grassy')
db_rosbag_path = db_loop.get('rosbag_path')

teach_rosbag_path = db_rosbag_path.get('teach')
# repeat_rosbag_path = db_rosbag_path.get('repeat1') # dont think this is needed

# for pose graph
pose_graph_path = db_loop.get('pose_graph_path').get('grassy')
print("pose graph path:",pose_graph_path)

db_bool = config['bool']
SAVE = db_bool.get('SAVE')
print("SAVE:",SAVE)
PLOT = db_bool.get('PLOT')

result_folder = config.get('output')
repeat = 1

print("list dir:",os.listdir(result_folder))
sorted_result_folder = sorted(os.listdir(result_folder),key=lambda x: int(re.search(r'repeat(\d+)', x).group(1)) if re.search(r'repeat(\d+)', x) else float('inf'))

print("sorted_result_folder:",sorted_result_folder)


# loop through every element in the result folder
for subfolder in sorted_result_folder:
    subfolder_path = os.path.join(result_folder, subfolder)  # Get the full path
    print(f"Subfolder: {subfolder}")
    print("Processing repeat: ",repeat)
    for item in sorted(os.listdir(subfolder_path)):
        item_path = os.path.join(subfolder_path, item)
        print(f"Item: {item}")
        # if an item is what I want I want to load it
        if item == f"grassy_repeat_{repeat}.npz":
            print(f"Item: {item}")
            data = np.load(item_path,allow_pickle=True)
            t_repeat=data["t_repeat"]
            ptr_repeat=data["ptr_repeat"]
            
            # reset t_repeat to start from 0
            # t_repeat = t_repeat - t_repeat[0]
            # now I want to plot the data
            if PLOT:
                plt.plot(t_repeat,ptr_repeat,label=f"path-tracking err repeat {repeat}",linewidth=0.5,alpha=0.8)
            
            print("t_repeat shape:", t_repeat.shape)
            print("ptr_repeat shape:", ptr_repeat.shape)

    repeat+=1
    if repeat == 3:
        break


fontsize = 12

localization_err_folder = os.path.join(result_folder, "ICRA_grassy_localization")
sorted_localization_err_folder = sorted(os.listdir(localization_err_folder),key=lambda x: int(re.search(r'localization_error_(\d+)', x).group(1)))
print("sorted_localization_err_folder:",sorted_localization_err_folder)

for loc_error_file in sorted_localization_err_folder:
    repeat_idx = sorted_localization_err_folder.index(loc_error_file)+1
    data = np.load(os.path.join(localization_err_folder,loc_error_file),allow_pickle=True)
    t=data["t"]
    # t = t - t[0] # reset t to start from 0
    dist = data["dist"]

    if PLOT:
        plt.plot(t,dist,label=f"localization err repeat {repeat_idx}",linewidth=1,marker = '*',markersize=7)

    # if repeat_idx == 5:
    #     break


if PLOT:
    plt.xlabel("Time (s)",fontsize=fontsize+5)
    plt.ylabel("Path tracking error (m)",fontsize=fontsize+5)
    plt.title(f"Path tracking and local error over time for grassy repeat",fontsize=fontsize+10)
    plt.legend(fontsize=fontsize)
    # plt.tight_layout()
    plt.grid()
    plt.show()

# the below uses the region metric
# now we can analyze the data 
# print("t_repeat shape:",t_repeat.shape)
# print("ptr_repeat shape:",ptr_repeat.shape)

regions_where_ptr_larger_than_20 = np.where(ptr_repeat>0.17)
# print("regions_where_ptr_larger_than_20:",regions_where_ptr_larger_than_20)

def find_continuous_regions(indices):
    """
    Finds continuous regions of indices in a given numpy array.

    Parameters:
        indices (numpy.ndarray): 1D numpy array of indices.

    Returns:
        List of tuples: Each tuple contains the start and end of a continuous region.
    """
    # Ensure the input is a numpy array
    indices = np.sort(np.array(indices))  # Ensure indices are sorted
    regions = []

    # Initialize start and previous index
    start = indices[0]
    prev = indices[0]

    for i in range(1, len(indices)):
        if indices[i] != prev + 1:  # Non-continuous point
            regions.append((start, prev))  # End the previous region
            start = indices[i]  # Start a new region
        prev = indices[i]

    # Append the final region
    regions.append((start, prev))
    return regions

regions = find_continuous_regions(regions_where_ptr_larger_than_20[0])
print("There are ",len(regions),"regions where the path tracking error is larger than 0.2")
print("regions:",regions)

paired = list(zip(t_repeat, ptr_repeat))

# now I need access to folder where the frames are stored
radar_folder = os.path.join(parent_folder,result_folder,"ICRA_grassy_repeat2/frames")
print("radar_folder:",radar_folder)

timestamps_list_images = []
for filename in os.listdir(radar_folder):
    if filename.endswith(".png"):
        match = re.search(r"(\d+\.\d+)", filename)
        if match:
            timestamps_list_images.append(float(match.group(1)))
            # timestamps_list_images.append((filename))

timestamps_list_images = sorted(timestamps_list_images)
# print("timestamps_list_images:",timestamps_list_images)

regions_folder = os.path.join(parent_folder,result_folder,"ICRA_grassy_repeat2/regions")
if(not os.path.exists(regions_folder)):
    os.makedirs(regions_folder)
    print(f"Folder '{regions_folder}' created.")

# now I want to find the corresponding timestamps for the regions
for region_of_interest in regions:
    print("region:",region_of_interest)
    region_number = regions.index(region_of_interest)
    subregion_folder= os.path.join(regions_folder,f"region_{region_number}")
    if(not os.path.exists(subregion_folder)):
        os.makedirs(subregion_folder)
        print(f"Folder '{subregion_folder}' created.")

    start_idx = region_of_interest[0]
    end_idx = region_of_interest[1]

    print("start_idx:",start_idx)
    print("end_idx:",end_idx)

    start_time = t_repeat[start_idx]
    end_time = t_repeat[end_idx]

    print("start_time:",start_time)
    print("end_time:",end_time)

    # now I want to find the corresponding images using closest time stamp
    start_closest_img_time = min(timestamps_list_images, key=lambda x:abs(x-start_time))
    end_closest_img_time = min(timestamps_list_images, key=lambda x:abs(x-end_time))

    print("start_closest_img_time:",start_closest_img_time)
    print("end_closest_img_time:",end_closest_img_time)

    # find the index of the closest image
    start_img_idx = timestamps_list_images.index(start_closest_img_time)
    end_img_idx = timestamps_list_images.index(end_closest_img_time)

    # I want to save everything from start_img_idx to end_img_idx into the region folder
    for i in range(start_img_idx,end_img_idx+1):
        img = cv2.imread(os.path.join(radar_folder,str(timestamps_list_images[i])+".png"))
        cv2.imwrite(os.path.join(subregion_folder,str(timestamps_list_images[i])+".png"),img)

    print(f"Region {regions.index(region_of_interest)}: start_img_idx:",start_img_idx)
    print(f"Region: {regions.index(region_of_interest)}: end_img_idx:",end_img_idx)

    # get the images
    start_img = cv2.imread(os.path.join(radar_folder,str(start_closest_img_time)+".png"))
    end_img = cv2.imread(os.path.join(radar_folder,str(end_closest_img_time)+".png"))

    # region_imgs_names = 

    # start_img = cv2.imread(os.path.join(radar_folder,str(start_time)+".png"))
    # end_img = cv2.imread(os.path.join(radar_folder,str(end_time)+".png"))

    # now I want to plot the images
    # fig, axs = plt.subplots(1,2)
    # axs[0].imshow(start_img)
    # axs[0].set_title(f"Start img time: {start_closest_img_time }")
    # axs[1].imshow(end_img)
    # axs[1].set_title(f"End img time: {end_closest_img_time}")
    # plt.show()







