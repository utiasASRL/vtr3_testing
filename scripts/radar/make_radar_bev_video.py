import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path
# import utm
from scipy.spatial.transform import Rotation, Slerp
from scipy import interpolate

from rosbags.typesys import get_types_from_msg, register_types, Stores, get_typestore
from rosbags.serde import serialize_cdr

import cv2
import cv_bridge

import matplotlib.pyplot as plt

import imageio

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

def get_radar_scan_images_and_timestamps(path):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(get_types_from_msg(FFT_MSG, 'nav_messages/msg/RadarFftDataMsg'))
    typestore.register(get_types_from_msg(SCAN_MSG,'navtech_msgs/msg/RadarBScanMsg'))

    # from rosbags.typesys.types import navtech_msgs__msg__RadarBScanMsg as RadarBScanMsg

    RadarBScanMsg = typestore.types['navtech_msgs/msg/RadarBScanMsg']
    scan_type = RadarBScanMsg.__msgtype__

    # intialize the arrays
    radar_times = []
    radar_images = []
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
            radar_time = round(radar_time,3)
            radar_times.append(radar_time)

            # now store the image
            bridge = cv_bridge.CvBridge()
            polar_img = bridge.imgmsg_to_cv2(msg.b_scan_img)
            fft_data = msg.b_scan_img.data.reshape((msg.b_scan_img.height, msg.b_scan_img.width))

            # print("fft_data",fft_data.shape)
            # plt.imshow(polar_img,cmap='gray', vmin=0, vmax=255)
            # plt.show()

            # print("polar image",polar_img.shape)
            print("encoder_values",msg.encoder_values)
            azimuths = (msg.encoder_values/16000)*2*np.pi
            print(azimuths)
            # print("azimuths",azimuths.shape)
            resolution = 0.040308  # change to your own resolution
            cart_resolution = 0.2384

            # convert the radar image to cartesian
            radar_image = radar_polar_to_cartesian(polar_img,azimuths, resolution, cart_resolution, 512)

            radar_images.append(radar_image)

            # break


    return radar_times, radar_images

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


# initlize the video writer
# Parameters for the video writer
output_video_path = './radar.avi'  # Output video file
frame_rate = 60.0  # Frames per second
frame_size = (512, 512)  # Frame size (width, height) of the video
codec = cv2.VideoWriter_fourcc(*'XVID')  # Video codec (XVID, MJPG, etc.)

# Create a VideoWriter object
video_writer = cv2.VideoWriter(output_video_path, codec, frame_rate, frame_size)

# to fill rosbag path 
radar_rosbag_path = "/home/samqiao/ASRL/vtr3_testing/localization_data/rosbags/grassy3" # fill this in
radar_times, radar_imgs = get_radar_scan_images_and_timestamps(radar_rosbag_path)

radar_times = np.array(radar_times)
radar_imgs = np.array(radar_imgs)

print(radar_imgs.shape)
print(radar_times.shape)
print("first radar time:",radar_times[0])
print("last radar time:",radar_times[-1])
radar_duration = radar_times[-1] - radar_times[0]
print("radar duration:",radar_duration)


# loop through the radar images and write them to the video
for radar_img in radar_imgs:
    # print("radar_img shape:",radar_img.shape)
    # Write the frame to the video
    radar_img = cv2.cvtColor(radar_img, cv2.COLOR_GRAY2BGR)
    video_writer.write(radar_img)

    # plt.imshow(radar_img,cmap='gray', vmin=0, vmax=255)
    # plt.show()


# Release the VideoWriter object   
video_writer.release()