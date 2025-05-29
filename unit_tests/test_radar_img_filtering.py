import rosbags
import cv2
from cv_bridge import CvBridge
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from matplotlib import pyplot as plt

import utm
import numpy as np

import sys
# Insert path at index 0 so it's searched first
sys.path.insert(0, "scripts")

from radar.utils.helper import *
from radar.utils.features import *
from radar.utils.helper import *

import yaml
import os

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
    db = config['radar_data']
    # Access database configuration
    db_loop = db.get('nov14')
    db_rosbag_path = db_loop.get('rosbag_path')

    teach_rosbag_path = db_rosbag_path.get('grassy_t4_k')

    # # for pose graph
    # pose_graph_path = db_loop.get('pose_graph_path').get('woody')
    # print("pose graph path:",pose_graph_path)

    db_bool = config['bool']
    SAVE = db_bool.get('SAVE')
    # print("SAVE:",SAVE)
    PLOT = db_bool.get('PLOT')

    result_folder = config.get('output')

    out_path_folder = os.path.join(result_folder,f"nov14_grassy_k_t1")
    if not os.path.exists(out_path_folder):
        os.makedirs(out_path_folder)
        print(f"Folder '{out_path_folder}' created.")
    else:
        print(f"Folder '{out_path_folder}' already exists.")    

    # change name here
    out_path = os.path.join(out_path_folder,f"k_strong_without_filtering.avi")

    # I want to plot them side by side actually one with filering one without
    fft_data,radar_timestamps,azimuth_angles, azimuth_timestamps_total,cart_imgs = get_radar_scan_images_and_timestamps(teach_rosbag_path)
    
    print("fft_data shape:", np.array(fft_data).shape)
    print("radar_timestamps shape:", np.array(radar_timestamps).shape)
    print("azimuth_angles shape:", np.array(azimuth_angles).shape)
    print("azimuth_timestamps_total shape:", np.array(azimuth_timestamps_total).shape)

    # need to convert azimuth angles into radians
    # azimuth_angles = azimuth_angles/16000 * 2 * np.pi

    radar_resolution = 0.040308
    index = 0
    warning_cnt = 0

    # intialize a video writer
    frame_rate = 30.0  # Frames per second
    # frame_size = (512, 512)  # Frame size (width, height) of the video
    frame_width, frame_height = 640, 480  # Example size; adjust based on your plots
    codec = cv2.VideoWriter_fourcc(*'XVID')  # Video codec (XVID, MJPG, etc.)

    out_video_path = os.path.join(out_path_folder,"modified_filtering.avi")

    # Create a VideoWriter object
    video_writer = cv2.VideoWriter(out_video_path, codec, frame_rate, (frame_width, frame_height))

    CAFAR = False


    for polar_img in fft_data:

        # index = fft_data.index(polar_img)
        print("index:",index)
        # need to create a filtered polar image 

        azimuths = azimuth_angles[index]
        # covert to radians azimuths
        azimuths = azimuths/16000 * 2 * np.pi

        print("azimuths shape:",azimuths.shape)

        azimuth_times = azimuth_timestamps_total[index]
        scan_stamp = radar_timestamps[index]
        print("scan_stamp:",scan_stamp)
        print("polar img shape:",polar_img.shape)

        cart_img = cart_imgs[index]
        print("cart img shape:",cart_img.shape)

        # without filtering
        if not CAFAR:
            k_strong_targets = KStrong(polar_img)

            print("k_strong_targets shape:",k_strong_targets.shape)
            cart_pts = polar_to_cartesian_points(azimuths, k_strong_targets,radar_resolution, downsample_rate=1)

            bev_pixels = convert_to_bev(cart_pts, 0.2384, 512)

            polar_targets = targets_to_polar_image(k_strong_targets,polar_img.shape)
        # print("polar_targets shape:",polar_targets.shape)

        # print("cart_pts shape:",cart_pts.shape)
        else:
            print("I am using modified CACFAR and the polar image shape is:",polar_img.shape)
            CACFAR_targets = modifiedCACFAR(polar_img)

            print("CACFAR_targets shape:",CACFAR_targets.shape)
            cart_pts = polar_to_cartesian_points(azimuths, CACFAR_targets,radar_resolution, downsample_rate=1)

            bev_pixels = convert_to_bev(cart_pts, 0.2384, 512)

            polar_targets = targets_to_polar_image(CACFAR_targets,polar_img.shape)

        

        print("bev_pixels shape:",bev_pixels.shape)

        cart_targets = radar_polar_to_cartesian(polar_targets, azimuths, radar_resolution, cart_resolution=0.2384, cart_pixel_width=512,
                             interpolate_crossover=False, fix_wobble=True)


        cart_img_combined = combine_cart_targets_on_cart_img(cart_img,bev_pixels,cart_targets)

        ######################################## with filtering ########################################################
        std = np.std(polar_img, axis=1)
        mean = np.mean(polar_img, axis=1)
        filtered_polar_img = (polar_img - mean[:, np.newaxis] - 2*std[:, np.newaxis]) * 2
        filtered_polar_img = np.clip(filtered_polar_img, 0, np.inf)

        # cv2.imshow("filtered_polar_img",filtered_polar_img)
        # cv2.imshow("polar_img",polar_img)

        if not CAFAR:

            k_strong_targets_filtered = KStrong(filtered_polar_img)

            polar_targets_filtered = targets_to_polar_image(k_strong_targets_filtered,filtered_polar_img.shape)

            if np.array_equal(k_strong_targets_filtered,k_strong_targets):
                print("index:",index)
                print("Warning! The filtered features are the same as the original features")
                warning_cnt+=1

            # cv2.imshow("polar_targets_filtered",polar_targets_filtered)

            cart_pts_filtered = polar_to_cartesian_points(azimuths, k_strong_targets_filtered,radar_resolution, downsample_rate=1)

            bev_pixels_filtered = convert_to_bev(cart_pts_filtered, 0.2384, 512)

            print("bev_pixels_filtered shape:",bev_pixels_filtered.shape)

            cart_targets_filtered = radar_polar_to_cartesian(polar_targets_filtered, azimuths, radar_resolution, cart_resolution=0.2384, cart_pixel_width=512,interpolate_crossover=False, fix_wobble=True) # shape 512 by 512
        
            cart_img_filtered = radar_polar_to_cartesian(filtered_polar_img, azimuths, radar_resolution, cart_resolution=0.2384, cart_pixel_width=512,interpolate_crossover=False, fix_wobble=True)
        else:
            CACFAR_targets_filtered = modifiedCACFAR(filtered_polar_img)

            polar_targets_filtered = targets_to_polar_image(CACFAR_targets_filtered,filtered_polar_img.shape)

            if np.array_equal(CACFAR_targets_filtered,CACFAR_targets):
                print("index:",index)
                print("Warning! The filtered features are the same as the original features")
                warning_cnt+=1
            
            # cv2.imshow("polar_targets_filtered",polar_targets_filtered)

            cart_pts_filtered = polar_to_cartesian_points(azimuths, CACFAR_targets_filtered,radar_resolution, downsample_rate=1)

            bev_pixels_filtered = convert_to_bev(cart_pts_filtered, 0.2384, 512)

            print("bev_pixels_filtered shape:",bev_pixels_filtered.shape)

            cart_targets_filtered = radar_polar_to_cartesian(polar_targets_filtered, azimuths, radar_resolution, cart_resolution=0.2384, cart_pixel_width=512,interpolate_crossover=False, fix_wobble=True) # shape 512 by 512

            cart_img_filtered = radar_polar_to_cartesian(filtered_polar_img, azimuths, radar_resolution, cart_resolution=0.2384, cart_pixel_width=512,interpolate_crossover=False, fix_wobble=True)
    
        

            # print("The filtered image exceed 255", np.where(cart_img_filtered > 255))
        # cv2.imshow("cart_img_filtered",cart_img_filtered)
        # cv2.imshow("cart_img",cart_img)
        # cv2.waitKey(0)

        # print("cart_img_filtered shape:",cart_img_filtered.shape)
        # print("cart_img_filtered type:",cart_img_filtered.dtype)

        cart_img_filtered = cart_img_filtered.astype(np.float32)
        
        cart_img_combined_filtered = combine_cart_targets_on_cart_img(cart_img_filtered,bev_pixels_filtered,cart_targets_filtered)

        frame_folder = os.path.join(out_path_folder,"frames")

        if not os.path.exists(frame_folder):
            os.makedirs(frame_folder)
            print(f"Folder '{frame_folder}' created.")

        frame_name = f"{scan_stamp}.png"

        fig, axs = plt.subplots(1, 2, tight_layout=True)
        # I want to print fig size

        axs[0].imshow(cart_img_combined)
        axs[0].set_title("Without Filtering")
        axs[1].imshow(cart_img_combined_filtered)
        axs[1].set_title("With Filtering")

        plt.savefig(os.path.join(frame_folder,frame_name))

        frame = cv2.imread(os.path.join(frame_folder,frame_name))

        video_writer.write(frame)

        index+=1

        # plt.show()

        break

    print("warning_cnt:",warning_cnt)
    print("out of ",len(fft_data))
    print("Done!")

    video_writer.release()




    


