import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path
import utm
from scipy.spatial.transform import Rotation, Slerp
from scipy import interpolate

from rosbags.typesys import get_types_from_msg, register_types, Stores, get_typestore
from rosbags.serde import serialize_cdr

import cv2
import cv_bridge

import matplotlib.pyplot as plt
import sys
# Insert path at index 0 so it's searched first
sys.path.insert(0, "scripts")
from radar.utils.path_comparison import*

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
def get_xyt_gps(path):
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


            # get the ros time
            gps_time_sec = msg.header.stamp.sec
            gps_time_nano_sec = msg.header.stamp.nanosec
            gps_time = gps_time_sec + gps_time_nano_sec/1e9
            t.append(gps_time)

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

def get_radar_scan_images_and_timestamps(path):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(get_types_from_msg(FFT_MSG, 'nav_messages/msg/RadarFftDataMsg'))
    typestore.register(get_types_from_msg(SCAN_MSG,'navtech_msgs/msg/RadarBScanMsg'))

    # from rosbags.typesys.types import navtech_msgs__msg__RadarBScanMsg as RadarBScanMsg

    RadarBScanMsg = typestore.types['navtech_msgs/msg/RadarBScanMsg']
    scan_type = RadarBScanMsg.__msgtype__

    # intialize the arrays
    radar_timestamps = []
    polar_images = []
    cart_images = []
    azimuth_angles = []
    azimuth_timestamps_total = []
    lookup_tb = dict()
    print("Processing: Getting image_timestamp and radar image")
    with AnyReader([Path(path)]) as reader:
        connections = [x for x in reader.connections if x.topic == '/radar_data/b_scan_msg']
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, scan_type)
            # need to make sure everything is in secs
            radar_time_sec = msg.b_scan_img.header.stamp.sec
            radar_time_nano_sec = msg.b_scan_img.header.stamp.nanosec
            radar_time = radar_time_sec + radar_time_nano_sec/1e9
            # round to 4 significant digits 
            # radar_time = round(radar_time,3)
            # this is the radar msg time to be used as file name
            radar_timestamps.append(radar_time)

            encoder_values = msg.encoder_values
            azimuth_angles.append(encoder_values)

            azimuth_timestamps = msg.timestamps
            azimuth_timestamps_total.append(azimuth_timestamps)

            # now store the image
            bridge = cv_bridge.CvBridge()
            polar_img = bridge.imgmsg_to_cv2(msg.b_scan_img)
            fft_data = msg.b_scan_img.data.reshape((msg.b_scan_img.height, msg.b_scan_img.width))

            polar_images.append(polar_img)

            # print("fft_data",fft_data.shape)
            # plt.imshow(polar_img,cmap='gray', vmin=0, vmax=255)
            # plt.show()

            # print("polar image",polar_img.shape)
            azimuths = msg.encoder_values/16000*2*np.pi
            # print("azimuths",azimuths.shape)
            resolution = 0.040308
            cart_resolution = 0.2384

            # convert the radar image to cartesian
            cart_image = radar_polar_to_cartesian(polar_img,azimuths, resolution, cart_resolution, 512)

            cart_images.append(cart_image)


    return polar_images,radar_timestamps, azimuth_angles, azimuth_timestamps_total, cart_images



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


def radar_polar_to_cartesian(fft_data, azimuths, radar_resolution, cart_resolution=0.2384, cart_pixel_width=640,
                             interpolate_crossover=False, fix_wobble=True):
    # TAKEN FROM PYBOREAS
    """Convert a polar radar scan to cartesian.
    Args:
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        fft_data (np.ndarray): Polar radar power readings
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        cart_resolution (float): Cartesian resolution (metres per pixel)
        cart_pixel_width (int): Width and height of the returned square cartesian output (pixels)
        interpolate_crossover (bool, optional): If true interpolates between the end and start  azimuth of the scan. In
            practice a scan before / after should be used but this prevents nan regions in the return cartesian form.

    Returns:
        np.ndarray: Cartesian radar power readings
    """
    # Compute the range (m) captured by pixels in cartesian scan
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    
    # Compute the value of each cartesian pixel, centered at 0
    coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)

    Y, X = np.meshgrid(coords, -1 * coords)
    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

    # Interpolate Radar Data Coordinates
    azimuth_step = (azimuths[-1] - azimuths[0]) / (azimuths.shape[0] - 1)
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution
    sample_v = (sample_angle - azimuths[0]) / azimuth_step
    # This fixes the wobble in the old CIR204 data from Boreas
    M = azimuths.shape[0]
    azms = azimuths.squeeze()
    if fix_wobble:
        c3 = np.searchsorted(azms, sample_angle.squeeze())
        c3[c3 == M] -= 1
        c2 = c3 - 1
        c2[c2 < 0] += 1
        a3 = azms[c3]
        diff = sample_angle.squeeze() - a3
        a2 = azms[c2]
        delta = diff * (diff < 0) * (c3 > 0) / (a3 - a2 + 1e-14)
        sample_v = (c3 + delta).astype(np.float32)

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    return cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR)


# ported from pt_errpr
def get_pose_path(path):
    x_gps,y_gps,t_gps = get_xyt_gps(path)
    z_gps = np.zeros_like(x_gps)
    pose_gps = np.vstack((x_gps,y_gps,z_gps)).T
    return pose_gps

def signed_distance_to_path(point: np.ndarray, path: np.ndarray):
    """Path is assumed to be formatted from path_to_matrix"""

    assert len(path.shape) == 2 and path.shape[1] == 7, "Matrix is ill formed. Should be nx7"
    assert len(point.shape) <= 2 and point.size == 3, "Point must be 3 dimensional"
    
    x0 = path[:, :3]
    tangent = path[:, 3:6]
    l = path[:, 6]
    points = np.zeros_like(x0)
    points[:] = point.T

    norm_vec = np.cross((points - x0), tangent)
    norm_dist = np.linalg.norm(norm_vec, axis=1)
    dists = np.linalg.norm((points - x0), axis=1)
    min_dist = np.min(dists)

    proj = np.diagonal(np.dot(points - x0, tangent.T))
    segment_filter = (proj > 0) & (proj < l)

    min_proj = 1e6

    #Handle the edge cases where points are beyond the limits of the path
    if np.sum(segment_filter) > 0:
        idx_min_proj = np.argmin(norm_dist[segment_filter])
        if norm_dist[segment_filter][idx_min_proj] < min_dist:
            return np.sign(norm_vec[segment_filter][idx_min_proj, 2]) * norm_dist[segment_filter][idx_min_proj]
    return min_dist


def RMSE(teach_path,repeat_path):
    x_teach,y_teach,t_teach = get_xyt_gps(teach_path)
    z_teach = np.zeros_like(x_teach)

    pose_teach = get_pose_path(teach_path)
    print("pose_teach shape:",pose_teach.shape)

    points_teach = path_to_matrix(pose_teach)
    print("points_teach shape:",points_teach.shape)

    x_repeat,y_repeat,t_repeat = get_xyt_gps(repeat_path)
    pose_repeat = get_pose_path(repeat_path)
    print("pose_repeat shape:",pose_repeat.shape)

    # distances_teach_repeat = distances_to_path(pose_repeat,points_teach)
    # hey I need to do a check here so that the error does not jump around (I also want to keep the time steps)
    previous_error = 0
    distances_teach_repeat = []
    for i in range(len(pose_repeat)):
        # print("i:",i)
        error = signed_distance_to_path(pose_repeat[i],points_teach)
        product = error*previous_error
        if product<0 and abs(error)>0.05 and abs(previous_error)>0.05:
            error = -1*error
        distances_teach_repeat.append(error)
        previous_error = error

    distances_teach_repeat = np.array(distances_teach_repeat)
    # # filter out the outliers
    # t_repeat = t_repeat[distances_teach_repeat < 1]
    # distances_teach_repeat = distances_teach_repeat[distances_teach_repeat < 1]


    Max = np.max(np.abs(distances_teach_repeat))
    RMSE = np.sqrt(np.mean(distances_teach_repeat**2))
    
    # fontsize = 20
    # plt.figure(figsize=(10,7))
    # plt.title("Repeat Individual Path-tracking error over time",fontsize=fontsize)
    # plt.scatter(t_repeat,distances_teach_repeat)
    # plt.xlabel('Time [s]')
    # # plt.axis('equal')
    # plt.legend([f"Path tracking error with mean {round(np.mean(distances_teach_repeat),3)} m, max {round(Max,3)} m"],fontsize=fontsize-5)
    # plt.grid()
    # plt.ylabel('Distance to TEACH path [m]')

    # plt.show()

    return round(RMSE,3),round(Max,4),distances_teach_repeat,t_repeat


def get_path_distance_from_gps(x_teach,y_teach):
    dist = 0
    for i in range(len(x_teach)-1):
        dist += np.sqrt((x_teach[i+1]-x_teach[i])**2 + (y_teach[i+1]-y_teach[i])**2)
    return dist




