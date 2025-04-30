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

# from radar.utils.helper import get_xyt_gps
from deps.path_tracking_error.fcns import *

from scipy import interpolate
import csv

import yaml



ROS_BAG_PATH = "/home/samqiao/ASRL/vtr3_data/routes/grassy/0913/grassy_t2"

# Load the rosbag
x_teach,y_teach,t_teach = get_xy_gt(ROS_BAG_PATH)

path_distance = get_path_distance_from_gps(x_teach,y_teach)

print("path_distance",path_distance)

plt.plot(x_teach,y_teach)
plt.axis('equal')

plt.show()