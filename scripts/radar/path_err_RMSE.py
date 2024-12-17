import rosbags
import cv2
from cv_bridge import CvBridge
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from matplotlib import pyplot as plt

import utm
import numpy as np
from utils.helper import *

import yaml
import os

import sys

# Insert path at index 0 so it's searched first
sys.path.insert(0, "scripts")

def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file.

    :param config_path: Path to the YAML configuration file.
    :return: A dictionary representing the configuration.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':
    # print current working directory
    print("Current working dir", Path.cwd())

    config = load_config('scripts/radar/config.yaml')
    # Access database configuration
    db = config['radar_data']
    db_rosbag_path = db.get('grassy_rosbag_path')

    db_bool = config['bool']
    SAVE = db_bool.get('SAVE')
    PLOT = db_bool.get('PLOT')

    result_folder = config.get('output')

    # change here
    repeat = 5

    # change here
    out_path_folder = os.path.join(result_folder,f"ICRA_grassy_repeat{repeat}/")
    if not os.path.exists(out_path_folder):
        os.makedirs(out_path_folder)
        print(f"Folder '{out_path_folder}' created.")
    else:
        print(f"Folder '{out_path_folder}' already exists.")    

    # change name here
    out_path = os.path.join(out_path_folder,f"grassy_repeat_{repeat}.npz")

    if SAVE:
        print("Saving the result to: ",out_path)

    # those are live gps bags
    # teach contains all the lidar and radar as well as gps
    # repeat bags are only gps
    teach_rosbag_path = db_rosbag_path.get('teach')
    repeat_rosbag_path = db_rosbag_path.get(f'repeat{repeat}') # change name here

    print("Processing repeat GPS bag path: ",repeat_rosbag_path)

    RMSE_repeat,MAX_repeat,ptr_repeat,t_repeat= RMSE(teach_rosbag_path,repeat_rosbag_path)

    print(f"RMSE_repeat_{repeat}", RMSE_repeat)
    print(f"MAX_repeat_{repeat}", MAX_repeat)
    # I want to plot path tracking error over time
    # reset t_repeat1 so it starts from 0
    # t_repeat1 = t_repeat1 - t_repeat1[0]

    if PLOT:
        for index in range(1,2,1):
            plt.plot(t_repeat, ptr_repeat)
            plt.grid()
            plt.xlabel('Time (s)')
            plt.ylabel('Path Tracking Etr (m)')
            plt.title(f"Repeat {repeat}: Path-tracking Error over time")
            plt.show()

    # I want to save all the result in npz format so I can load quickly later

    if SAVE:
        np.savez_compressed(out_path,ptr_repeat=ptr_repeat,t_repeat=t_repeat,RMSE_repeat=RMSE_repeat,MAX_repeat=MAX_repeat)

 