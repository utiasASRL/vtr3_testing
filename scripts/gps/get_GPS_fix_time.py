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

# Import your custom message type
from novatel_oem7_msgs.msg import Oem7RawMsg
from sensor_msgs.msg import NavSatFix # for GPS

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

def wrap_angle(theta):
    if(theta > np.pi):
        return theta - 2*np.pi
    elif(theta < -np.pi):
        return theta + 2*np.pi
    else:
        return theta


def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix SO(3) to Euler angles (roll, pitch, yaw).

    Parameters:
        R (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
        tuple: A tuple containing roll, pitch, and yaw angles (in radians).
    """
    assert R.shape == (3, 3), "Input rotation matrix must be 3x3."

    # Check if the matrix is a valid rotation matrix
    if not np.allclose(np.dot(R.T, R), np.eye(3)) or not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError("Input matrix is not a valid rotation matrix.")

    # Extract the Euler angles
    pitch = -np.arcsin(R[2, 0])

    if np.isclose(np.cos(pitch), 0):
        # Gimbal lock case
        roll = 0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return roll, pitch, yaw

if __name__ == '__main__':
    
    config = load_config(os.path.join(parent_folder,'scripts/config.yaml'))

    # Access database configuration
    db = config['radar_data']
    db_loop = db.get('new_rss_parking')
    db_rosbag_path = db_loop.get('rosbag_path')

    # print("db_rosbag_path",db_rosbag_path)

    # for pose graph
    trial = 't6'

    teach_rosbag_path = db_rosbag_path.get('parking_'+trial)
    # repeat_rosbag_path = db_rosbag_path.get('repeat1') # dont think this is needed

    
    # pose_graph_path = db_loop.get('pose_graph_path').get('grassy_t1')
    # print("pose graph path:",pose_graph_path)

    db_bool = config['bool']
    SAVE = db_bool.get('SAVE')
    # print("SAVE:",SAVE)
    PLOT = db_bool.get('PLOT')

    result_folder = config.get('output')

    save_folder = os.path.join(result_folder, "jan_29_rss_parking_"+trial)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Folder '{save_folder}' created.")

    # typestore = get_typestore(Stores.ROS2_HUMBLE)
    # oem7Msg = typestore.types['novatel_oem7_msgs/msg/Oem7RawMsg']
    # scan_type = oem7Msg.__msgtype__

    cnt = 0
    rclpy.init()

    # Create a SequentialReader
    reader = SequentialReader()

    # Prepare storage options (SQLite is the default for ROS2)
    storage_options = StorageOptions(
        uri=teach_rosbag_path,
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
    raw_topic = '/novatel/oem7/oem7raw'
    fix_topic = '/novatel/oem7/fix'

    raw_cnt = 0
    fix_cnt = 0
    fix_data = []
    while reader.has_next():
        (topic, data, t) = reader.read_next()

        if topic == raw_topic:
            # Deserialize Oem7RawMsg
            msg = deserialize_message(data, Oem7RawMsg)

            # Access fields in Oem7RawMsg
            gps_ros_time = (msg.header.stamp.sec +
                            msg.header.stamp.nanosec / 1e9)
            print(f"[RAW] | GPS Time: {gps_ros_time}")
            raw_cnt += 1

        elif topic == fix_topic:
            # Deserialize NavSatFix (or your actual fix-type message)
            msg = deserialize_message(data, NavSatFix)
            gps_fix_ros_time = (msg.header.stamp.sec +
                            msg.header.stamp.nanosec / 1e9)
            # Access fields in NavSatFix
            # e.g. latitude, longitude, altitude
            print(f"[FIX] Timestamp(ns): {gps_fix_ros_time} | Lat: {msg.latitude}, Lon: {msg.longitude}")
            fix = {}
            fix['timestamp'] = gps_fix_ros_time
            fix['latitude'] = msg.latitude
            fix['longitude'] = msg.longitude

            fix_data.append(fix)
            fix_cnt += 1

    # Print summary
    print(f"Total RAW messages read: {raw_cnt}")
    print(f"Total FIX messages read: {fix_cnt}")


    # I want to write fix data line by line to a csv file
    with open(os.path.join(save_folder,f'jan_grassy_{trial}_gps_fix.csv'), 'w') as f:
        f.write("Timestamp(ns),Latitude,Longitude\n")
        for fix in fix_data:
            f.write(f"{fix['timestamp']},{fix['latitude']},{fix['longitude']}\n")




    # Shutdown rclpy
    rclpy.shutdown()