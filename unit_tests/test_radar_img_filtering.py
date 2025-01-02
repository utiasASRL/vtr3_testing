import rosbags
import cv2
from cv_bridge import CvBridge
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from matplotlib import pyplot as plt

import utm
import numpy as np

# Insert path at index 0 so it's searched first
sys.path.insert(0, "scripts")

from scripts.radar.utils.helper import *

import yaml
import os

import sys

# Insert path at index 0 so it's searched first
sys.path.insert(0, "scripts")

# this script will take in the 1114 data and output frame by frame images

# two experiments
# modified-CACFAR with or without filtering

# k strongest points with or without filtering

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

    config = load_config('scripts/config.yaml')
    # Access database configuration
    db = config['radar_data']
    db_rosbag_path = db.get('grassy_rosbag_path')

    db_bool = config['bool']
    SAVE = db_bool.get('SAVE')
    PLOT = db_bool.get('PLOT')

    result_folder = config.get('output')

    
    out_path_folder = os.path.join(result_folder,f"nov14_grassy_k_t1")
    if not os.path.exists(out_path_folder):
        os.makedirs(out_path_folder)
        print(f"Folder '{out_path_folder}' created.")
    else:
        print(f"Folder '{out_path_folder}' already exists.")    

    # change name here
    out_path = os.path.join(out_path_folder,f"grassy_repeat_{repeat}.")


