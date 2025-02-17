import os
import matplotlib.pyplot as plt
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
# from vtr_pose_graph.graph_iterators import PriviledgedIterator
# import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse

from radar.utils.helper import *

# from rosbags.typesys import get_types_from_msg, register_types, Stores, get_typestore
# from rosbags.serde import serialize_cdr

import yaml

import sys

import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

from sensor_msgs.msg import Imu


# print current working directory
print("Current working dir", os.getcwd())

# Insert path at index 0 so it's searched first
sys.path.insert(0, "scripts")

from radar.utils.helper import get_xyt_gps

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

rosbag_path = "/home/samqiao/ASRL/vtr3_data/routes/parking/0911/parking_t1" #ICRA data

rosbag_path = "/home/samqiao/ASRL/vtr3_data/250128/parking_1" #new parking data

# rosbag_path = "/home/samqiao/ASRL/vtr3_data/250128/parking_2" #new parking data

rosbag_path = "/home/samqiao/ASRL/vtr3_data/rss_routes/250114/mars_t2" #mars_t1

rosbag_path = "/home/samqiao/ASRL/vtr3_data/rss_routes/woody_t4" #woody_t2

rosbag_path = "/home/samqiao/ASRL/vtr3_data/new_rss_routes/grassy/grassy5" #grassy_t2

rosbag_path = "/home/samqiao/ASRL/vtr3_data/250128/parking_2" #0128 parking

rosbag_path = "/home/samqiao/ASRL/vtr3_data/new_rss_routes/parking_bad_data_not_usable/parking5" #parking5

rosbag_path = "/home/samqiao/ASRL/vtr3_data/routes/parking/0911/parking_t1" #ICRA parking_t1

rosbag_path = "/home/samqiao/ASRL/vtr3_data/routes/mars/0907/mars_t1" #mars_t1

# # lets try a grassy loop and see what is going on
# rosbag_path = "/home/samqiao/ASRL/vtr3_data/new_rss_routes/grassy/grassy2" # new grassy data

print("rosbag path", rosbag_path)


cnt = 0
rclpy.init()

# Create a SequentialReader
reader = SequentialReader()

# Prepare storage options (SQLite is the default for ROS2)
storage_options = StorageOptions(
    uri=rosbag_path,
    storage_id='sqlite3'
)

converter_options = ConverterOptions('', '')

# Open the bag
reader.open(storage_options, converter_options)

# List available topics
topic_types = reader.get_all_topics_and_types()
print("Available topics in bag:")
for t in topic_types:
    print(f" - {t.name} (type: {t.type})")

# Define the topic you want to read
imu_topic = '/ouster/imu'

raw_cnt = 0
fix_data = []

ax_data = []
ay_data = []
az_data = []
wx_data = []
wy_data = []
wz_data = []
orientation_data = []
rostime = []

while reader.has_next():
    (topic, data, t) = reader.read_next()

    if topic == imu_topic:
        print(f"Reading message {raw_cnt} from topic {topic}")
        # Deserialize imu
        msg = deserialize_message(data, Imu)

        # Access fields in IMU
        ouster_ros_time = (msg.header.stamp.sec +
                        msg.header.stamp.nanosec / 1e9)
        # print(f"[Ouster IMU] | Time: {ouster_ros_time}")

        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        ax_data.append(ax)
        ay_data.append(ay)
        az_data.append(az)

        # print(f"Linear Acceleration: {ax}, {ay}, {az}")


        wx = msg.angular_velocity.x
        wy = msg.angular_velocity.y
        wz = msg.angular_velocity.z

        wx_data.append(wx)
        wy_data.append(wy)
        wz_data.append(wz)

        # print(f"Angular Velocity: {wx}, {wy}, {wz}")

        orientation_x = msg.orientation.x
        orientation_y = msg.orientation.y
        orientation_z = msg.orientation.z
        orientation_w = msg.orientation.w

        orientation_data.append([orientation_x, orientation_y, orientation_z, orientation_w])
        rostime.append(ouster_ros_time)

        raw_cnt += 1

# the total number of messages read
# Print summary 
print(f"Total Ouster IMU messages read: {raw_cnt}")   

# Shutdown rclpy
rclpy.shutdown()



rostime = np.array(rostime)
rostime = np.squeeze(rostime - rostime[0])

# wz first second
print("wz first second mean", np.mean(wz_data[0:100]))

# lets plot the 6-axis velocity data
# Create subplots
fig, axes = plt.subplots(nrows=6, sharex=True, figsize=(8, 10))
fig.suptitle("Velocity and Angular Velocity over Time")

# Plot data
axes[0].plot(rostime, ax_data, label="ax", color="r", linewidth=0.1)
axes[1].plot(rostime, ay_data, label="ay", color="g", linewidth=0.1)
axes[2].plot(rostime, az_data, label="az", color="b", linewidth=0.1)
axes[3].plot(rostime, wx_data, label="wx", color="c", linewidth=0.1)
axes[4].plot(rostime, wy_data, label="wy", color="m", linewidth=0.1)
axes[5].plot(rostime, wz_data, label="wz", color="y", linewidth=0.1)

# I also want to plot all the mean values of the velocity
# Calculate mean values
ax_mean = np.mean(ax_data)
ay_mean = np.mean(ay_data)
az_mean = np.mean(az_data)
wx_mean = np.mean(wx_data)
wy_mean = np.mean(wy_data)
wz_mean = np.mean(wz_data)

# Plot mean values
axes[0].axhline(ax_mean, color="r", linestyle="--", label=f"ax mean: {ax_mean:.2f}")
axes[1].axhline(ay_mean, color="g", linestyle="--", label=f"ay mean: {ay_mean:.2f}")
axes[2].axhline(az_mean, color="b", linestyle="--", label=f"az mean: {az_mean:.2f}")
axes[3].axhline(wx_mean, color="c", linestyle="--", label=f"wx mean: {wx_mean:.2f}")
axes[4].axhline(wy_mean, color="m", linestyle="--", label=f"wy mean: {wy_mean:.2f}")
axes[5].axhline(wz_mean, color="y", linestyle="--", label=f"wz mean: {wz_mean:.4f}")


# Set labels
axes[5].set_xlabel("Time (s)")
for i, ax in enumerate(axes):
    ax.set_ylabel(["ax", "ay", "az", "wx", "wy", "wz"][i])
    ax.legend()
    ax.grid(True)


# a second figure just for the gyro
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(rostime, wz_data, label="wz", color="purple", linewidth=0.05, marker='o',markersize=0.1)
ax.axhline(wz_mean, color="purple", linestyle="--", label=f"wz mean: {wz_mean:.4f}")

# Labels and title
ax.set_title("Gyroscope Wz Over Time (Detailed View)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angular Velocity Wz (rad/s)")

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
