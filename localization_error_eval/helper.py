import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path
import utm
from scipy.spatial.transform import Rotation, Slerp
from scipy import interpolate

from rosbags.typesys import get_types_from_msg, register_types, Stores, get_typestore
from rosbags.serde import serialize_cdr

# I will need to register the custom navtech message type
# define messages
SCAN_MSG = """
# A ROS message carrying a B Scan and its associated metadata (e.g. timestamps, encoder IDs)
# B Scan from one rotation of the radar, also holds the time stamp information
sensor_msgs/Image b_scan_img

# The encoder values encompassed by the b scan
uint16[] encoder_values

# The timestamps of each azimuth in the scan
uint64[] timestamps
"""

FFT_MSG = """
# A ROS message based on an FFT data message from a radar Network order means big endian

# add a header message to hold message timestamp
std_msgs/Header header

# angle (double) represented as a network order (uint8_t) byte array (don't use)
uint8[] angle

# azimuth (uint16_t) represented as a network order (uint8_t) byte array (encoder tick number)
uint8[] azimuth

# sweep_counter (uint16_t) represented as a network order (uint8_t) byte array
uint8[] sweep_counter

# ntp_seconds (uint32_t) represented as a network order (uint8_t) byte array
uint8[] ntp_seconds

# ntp_split_seconds (uint32_t) represented as a network order (uint8_t) byte array
uint8[] ntp_split_seconds

# data (uint8_t) represented as a network order (uint8_t) byte array
uint8[] data

# data_length (uint16_t) represented as a network order (uint8_t) byte array
uint8[] data_length """



def calculate_Euclidian_Distance(p1, p2):
    """
    Calculate the Euclidian distance between two points
    :param p1: Point 1
    :param p2: Point 2
    :return: Euclidian distance
    """
    return np.linalg.norm(p1 - p2)


def get_inverse_tf(T):
  """Returns the inverse of a given 4x4 homogeneous transform.
    Args:
        T (np.ndarray): 4x4 transformation matrix
    Returns:
        np.ndarray: inv(T)
    """
  T2 = T.copy()
  T2[:3, :3] = T2[:3, :3].transpose()
  T2[:3, 3:] = -1 * T2[:3, :3] @ T2[:3, 3:]
  return T2


def get_estimated_error(path_teach, path_repeat):
    """
    Calculate the estimated error between two paths
    :param path_teach: Teach path
    :param path_repeat: Repeat path
    :return: Estimated error
    """
    # Calculate the Euclidian distance between each point in the teach path and the repeat path
    distances = []
    for i in range(len(path_teach[0])):
        t_teach = path_teach[2, i]
        t_repeat = path_repeat[2, i]
        # print("teach_timestamp, repeat_timestamp",t_teach, t_repeat)
        distances.append([calculate_Euclidian_Distance(path_teach[:2, i], path_repeat[:2, i]), t_teach, t_repeat])

    # Calculate the mean of the Euclidian distances
    return np.array(distances)



# create reader instance and open for reading
def get_xy_gps(path):
    x_total = []
    y_total = []
    # z_total = []
    t = []

    cnt = 0
    with AnyReader([Path(path)]) as reader:
        connections = [x for x in reader.connections if x.topic == '/novatel/fix']
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            # print(msg)
            longtitude = msg.longitude

            latitude = msg.latitude

            # x,y,z = get_cartesian(lat=latitude,lon=longtitude)
            x,y = utm.from_latlon(latitude, longtitude)[:2]

            if cnt == 0:
                x_offset = x
                y_offset = y
                t_offset = timestamp
            
            x_total.append(x)
            y_total.append(y)
            t.append((timestamp/ 1e9))

            # z_total.append(z)
            cnt += 1
    return np.array(x_total),np.array(y_total),np.array(t)


def get_radar_time_ros_time_lookup(path):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(get_types_from_msg(FFT_MSG, 'nav_messages/msg/RadarFftDataMsg'))
    typestore.register(get_types_from_msg(SCAN_MSG,'navtech_msgs/msg/RadarBScanMsg'))

    # from rosbags.typesys.types import navtech_msgs__msg__RadarBScanMsg as RadarBScanMsg

    RadarBScanMsg = typestore.types['navtech_msgs/msg/RadarBScanMsg']
    scan_type = RadarBScanMsg.__msgtype__

    # intialize the arrays
    radar_times = []
    ros_times = []
    lookup_tb = dict()
    print("Processing: Getting ros_time and radar image time lookup table")
    with AnyReader([Path(path)]) as reader:
        connections = [x for x in reader.connections if x.topic == '/radar_data/b_scan_msg']
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, scan_type)
            # need to make sure everything is in secs
            radar_time_sec = msg.b_scan_img.header.stamp.sec
            radar_time_nano_sec = msg.b_scan_img.header.stamp.nanosec
            radar_time = radar_time_sec + radar_time_nano_sec/1e9
            # round to 4 significant digits 
            radar_time = round(radar_time,3)
            radar_times.append(radar_time)

            # the ros time is also in secs
            ros_time = timestamp/1e9
            # round to 4 significant digits 
            ros_time = round(ros_time,3)
            ros_times.append(ros_time)
            lookup_tb[radar_time] = ros_time
            print(f"Associating ros time: {ros_time} with radar image time {radar_time}")

    return radar_times, ros_times, lookup_tb


def linear_interpolate(x1,y1,t1,x2,y2,t2):
    #synchronize t2 onto t1
    t1 = np.squeeze(t1)
    t2 = np.squeeze(t2)
    # print("sam debug",t1)
    # print("sam debug",t2)
    x2_synced = np.interp(t1,t2,np.squeeze(x2))
    y2_synced = np.interp(t1,t2,np.squeeze(y2))
    # print("x2_synced:",x2_synced)

    return x2_synced,y2_synced


def get_closest_timestamp(t_test,t_array):
    t_array = np.squeeze(t_array)
    idx = (np.abs(t_array - t_test)).argmin()
    return t_array[idx]



