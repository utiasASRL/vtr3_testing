import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path
import utm
from scipy.spatial.transform import Rotation, Slerp
from scipy import interpolate

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
    radar_times = []
    ros_times = []
    lookup_tb = dict()
    with AnyReader([Path(path)]) as reader:
        connections = [x for x in reader.connections if x.topic == '/radar_data/b_scan_msg']
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            radar_time = msg.header.stamp
            ros_time = timestamp/1e9
            lookup_tb[radar_time] = ros_time

    return lookup_tb


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
