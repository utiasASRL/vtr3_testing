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

config = load_config(os.path.join(parent_folder,'scripts/radar/config.yaml'))

# Access database configuration
db = config['radar_data']
db_rosbag_path = db.get('grassy_rosbag_path')

teach_rosbag_path = db_rosbag_path.get('teach')
repeat_rosbag_path = db_rosbag_path.get('repeat1') # dont think this is needed

# for pose graph
pose_graph_path = db.get('pose_graph_path').get('grassy')
print("pose graph path:",pose_graph_path)

db_bool = config['bool']
SAVE = db_bool.get('SAVE')
print("SAVE:",SAVE)
PLOT = db_bool.get('PLOT')

result_folder = config.get('output')





