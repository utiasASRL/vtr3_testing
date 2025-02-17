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



teach_ppk_path = "localization_data/ppk/grassy_t2/grassy_t2.txt"
teach_raw_path = "localization_data/ppk/grassy_t2/grassy2_0_BESTPOS.ASCII"

repeat_ppk_path = "localization_data/ppk/grassy_t3/grassy_t3.txt"
repeat_raw_path = "localization_data/ppk/grassy_t3/grassy3_0_BESTPOS.ASCII"

x_ppk_teach, y_ppk_teach = read_PPK_file(teach_ppk_path)
x_raw_teach, y_raw_teach = read_ASCII_file(teach_raw_path)

x_ppk_repeat, y_ppk_repeat = read_PPK_file(repeat_ppk_path)
x_raw_repeat, y_raw_repeat = read_ASCII_file(repeat_raw_path)

fontsize = 20
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')


plt.figure()
# # teach path
plt.scatter(x_ppk_teach, y_ppk_teach, label='teach ppk f grassy 2',marker='*',s=3)
# plt.scatter(x_simulated_teach,y_simulated_teach, label='teach simulated rtk f',marker='*',s=3)
plt.scatter(x_ppk_repeat,y_ppk_repeat, label='repeat ppk f grassy 3',marker='*',s=3)

# plt.scatter(x_raw_teach, y_raw_teach, label='teach raw rtk f',marker='*',s=3)
# plt.scatter(x_raw_repeat, y_raw_repeat, label='repeat raw rtk f',marker='*',s=3)



plt.axis('equal')
plt.legend(fontsize=fontsize)
plt.xlabel('x (m)',fontsize=fontsize)
plt.ylabel('y (m)',fontsize=fontsize)
plt.xticks(fontsize=fontsize-5)
plt.yticks(fontsize=fontsize-5)
plt.title('PPK comparison',fontsize=fontsize)
plt.grid(True)

plt.show()

