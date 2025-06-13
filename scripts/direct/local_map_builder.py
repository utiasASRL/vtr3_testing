import yaml
import torch
import torchvision
import os
import numpy as np

from pylgmath import Transformation
import cv2
import pyboreas as pb

import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode

parent_folder = "/home/samqiao/ASRL/vtr3_testing"

T_radar_robot =  Transformation(T_ba = np.array([[1.000, 0.000, 0.000, 0.025],
                                                 [0.000, -1.000 , 0.000, -0.002],
                                                 [0.000 ,0.000, -1.000 , 1.032],
                                                 [0.000 , 0.000 ,0.000, 1.000]]))


# Leo's comments: for each teach pose, I'm gonna compute the n nearest poses (the distance in some bounds).
# with the relative translormation to the current pose I'm gonna trasform the cartesian point in the current pose
# for each polar image compute the cartesian point, apply the relative trasform, project back

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
    # print("in radar_polar_to_cartesian")
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

    # print("------")
    # print("sample_angle.shape",sample_angle.shape)
    # print("azimuths[0]",azimuths[0])
    # print("azimuth step shape" ,azimuth_step.shape)

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

def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file.

    :param config_path: Path to the YAML configuration file.
    :return: A dictionary representing the configuration.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def cartToLocalMapID(xy, local_map_res, local_map_zero_idx):
    out = torch.empty_like(xy, device=device)
    out[:,:,0,0] = (xy[:,:,0,0] / (-local_map_res)) + local_map_zero_idx
    out[:,:,1,0] = (xy[:,:,1,0] / (local_map_res)) + local_map_zero_idx
    return out

def localMapToPolarCoord(local_map_xy):
    with torch.no_grad():
        # Get the polar coordinates of the image
        cart = local_map_xy

        # Get the new polar coordinates
        polar = torch.zeros((cart.shape[0], cart.shape[1], 2)).to(device)
        polar[:, :, 0] = torch.atan2(cart[:, :, 1, 0], cart[:, :, 0, 0])
        polar[:, :, 1] = torch.sqrt(cart[:, :, 0, 0]**2 + cart[:, :, 1, 0]**2)

        #polar[polar[:,:,0]<0] = polar[polar[:,:,0]<0] + torch.tensor((2*torch.pi, 0)).to(self.device)
        #polar[:,:,0] *= (nb_azimuths / (2*torch.pi))
        #polar[:,:,1] /= radar_res
        return polar

def bilinearInterpolation(im, az_r):
    with torch.no_grad():
        az0 = torch.floor(az_r[:, :, 0]).int()
        az1 = az0 + 1
        
        r0 = torch.floor(az_r[:, :, 1]).int()
        r1 = r0 + 1

        az0 = torch.clamp(az0, 0, im.shape[0]-1)
        az1 = torch.clamp(az1, 0, im.shape[0]-1)
        r0 = torch.clamp(r0, 0, im.shape[1]-1)
        r1 = torch.clamp(r1, 0, im.shape[1]-1)
        az_r[:,:,0] = torch.clamp(az_r[:,:,0], 0, im.shape[0]-1)
        az_r[:,:,1] = torch.clamp(az_r[:,:,1], 0, im.shape[1]-1)
        
        Ia = im[ az0, r0 ]
        Ib = im[ az1, r0 ]
        Ic = im[ az0, r1 ]
        Id = im[ az1, r1 ]
        
        local_1_minus_r = (r1.float()-az_r[:, :, 1])
        local_r = (az_r[:, :, 1]-r0.float())
        local_1_minus_az = (az1.float()-az_r[:, :, 0])
        local_az = (az_r[:, :, 0]-az0.float())
        wa = local_1_minus_az * local_1_minus_r
        wb = local_az * local_1_minus_r
        wc = local_1_minus_az * local_r
        wd = local_az * local_r

        img_interp = wa*Ia + wb*Ib + wc*Ic + wd*Id

        return img_interp

def moveLocalMap(pos, rot, local_map, local_map_xy, local_map_res, local_map_zero_idx):
    with torch.no_grad():
        # Set to zero the first and last row and column of the localMap
        local_map[0, :] = 0
        local_map[-1, :] = 0
        local_map[:, 0] = 0
        local_map[:, -1] = 0

        # Get the coordinate of the new localMap in the former localMap
        temp_rot_mat = torch.tensor([[torch.cos(rot), -torch.sin(rot)], [torch.sin(rot), torch.cos(rot)]]).to(device)
        temp_pos = pos.reshape((-1,1))

        # Get the new coordinates
        new_xy = temp_rot_mat @ local_map_xy + temp_pos
        new_idx = cartToLocalMapID(new_xy, local_map_res, local_map_zero_idx)

        # Get the new localMap via bilinear interpolation
        local_map = bilinearInterpolation(local_map, new_idx).squeeze().float()
        return local_map

config_warthog = load_config(os.path.join(parent_folder,'scripts/direct/warthog_config.yaml'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Prepare the local_map cart coordinates
radar_res = 0.040308
local_map_res = float(config_warthog['direct']['local_map_res'])
max_local_map_range = float(config_warthog['direct']['max_local_map_range'])
local_map_size = int(max_local_map_range/local_map_res)*2 + 1

temp_x = (torch.arange( -local_map_size//2, local_map_size//2, 1).to(device) + 1) * local_map_res

X = -temp_x.unsqueeze(0).T.repeat(1,local_map_size)
Y = temp_x.unsqueeze(0).repeat(local_map_size,1)
local_map_xy = torch.stack((X, Y), dim=2).unsqueeze(-1).to(device)

local_map_res = torch.tensor(local_map_res).to(device)
local_map_zero_idx = torch.tensor(int(max_local_map_range/local_map_res)).to(device)

local_map_polar = localMapToPolarCoord(local_map_xy)

# change here
config = load_config(os.path.join(parent_folder,'scripts/direct/direct_configs/direct_config_sam.yaml'))
result_folder = config.get('output')
out_path_folder = os.path.join(result_folder,f"grassy_t2_r3/")
if not os.path.exists(out_path_folder):
    os.makedirs(out_path_folder)
    print(f"Folder '{out_path_folder}' created.")
else:
    print(f"Folder '{out_path_folder}' already exists.")    

TEACH_FOLDER = os.path.join(out_path_folder, "teach")

teach_df = np.load(os.path.join(TEACH_FOLDER, "teach.npz"),allow_pickle=True)
# in the teach
# 1. (932,400,1712) images
teach_polar_imgs = teach_df['teach_polar_imgs']
# 2. (932,400, 1) azimuth angles
teach_azimuth_angles = teach_df['teach_azimuth_angles']
# 3. (932,400, 1) azimuth timestamps
teach_azimuth_timestamps = teach_df['teach_azimuth_timestamps']
# 4. (932,1) vertex timestamps
teach_vertex_timestamps = teach_df['teach_vertex_timestamps']
# 5. Pose at each vertex: (932,4,4)
teach_vertex_transforms = teach_df['teach_vertex_transforms']
# 6. teach vertext time
teach_times = teach_df['teach_times']


max_distance = 1 # 2 m 

# save path for local maps
local_map_path = "/home/samqiao/ASRL/vtr3_testing/scripts/direct/grassy_t2_r3"
local_map_path = local_map_path + '/local_map_vtr/'
os.makedirs(local_map_path, exist_ok=True)

cnt = 0 

for index in range(len(teach_times)): # for every pose
    cur_pose_time = teach_times[index]

    cur_pose = list(teach_vertex_transforms[index][0].values())[0].matrix() # T_robot_world

    local_map = torch.zeros((local_map_size, local_map_size)).to(device)

    for i in range(len(teach_times)): # for the neighbouring poses
        print(f"Processing pose {i} at time {teach_times[i][0]}")
        # if i == index:
            # continue
        pose = list(teach_vertex_transforms[i][0].values())[0].matrix()
        # compute the distance between the poses
        # print(type(pose))
        delta_pose = np.linalg.inv(cur_pose) @ pose  # T_radar_robot @ delta_pose @ T_radar_robot.inverse().matrix() # need to do a transofrm

        delta_pose = T_radar_robot.matrix() @ delta_pose @ np.linalg.inv(T_radar_robot.matrix())  # T_radar_robot @ delta_pose @ T_radar_robot.inverse().matrix
        
        if np.linalg.norm(delta_pose[0:3, 3]) > max_distance:
            continue

        azimuths = torch.tensor(teach_azimuth_angles[i]).to(device).float()
        nb_azimuths = torch.tensor(len(azimuths)).to(device)
        polar_intensity = torch.tensor(teach_polar_imgs[i]).to(device).float()

        # add normalization to polar_image
        # polar_intensity = torch.tensor(polar_image).to(device)
        polar_std = torch.std(polar_intensity, dim=1)
        polar_mean = torch.mean(polar_intensity, dim=1)
        polar_intensity -= (polar_mean.unsqueeze(1) + 2*polar_std.unsqueeze(1))
        polar_intensity[polar_intensity < 0] = 0
        polar_intensity = torchvision.transforms.functional.gaussian_blur(polar_intensity.unsqueeze(0), (9,1), 3).squeeze()
        polar_intensity /= torch.max(polar_intensity, dim=1, keepdim=True)[0]
        polar_intensity[torch.isnan(polar_intensity)] = 0


        # if i == index:
        #     print("haha")
        #     cart_resolution = 0.234
        #     cart_pixel_width = 640
        #     cart_polar_intensity = pb.utils.radar.radar_polar_to_cartesian(teach_azimuth_angles[i].astype(np.float32), polar_intensity.cpu().numpy(), radar_res, 0.234, 640, False, True)
            # cv2.imshow("cart_polar_intensity", cart_polar_intensity)
            # cv2.imwrite(local_map_path + "teach_scan" + str(cur_pose_time[0]) + '.png', cart_polar_intensity)


        temp_polar_to_interp = local_map_polar.clone()            
        temp_polar_to_interp[:,:,0] -= (azimuths[0])
        temp_polar_to_interp[temp_polar_to_interp[:,:,0]<0] = temp_polar_to_interp[temp_polar_to_interp[:,:,0]<0] + torch.tensor((2*torch.pi, 0)).to(device)
        temp_polar_to_interp[:,:,0] *= ((nb_azimuths) / (2*torch.pi))
        temp_polar_to_interp[:,:,1] -= (radar_res/2.0)
        temp_polar_to_interp[:,:,1] /= radar_res

        cart_img = bilinearInterpolation(polar_intensity, temp_polar_to_interp)
        
        position = torch.tensor(delta_pose[0:2, 3]).to(device).float()
        rotation = torch.atan2(torch.tensor(delta_pose[1, 0]).to(device).float(), torch.tensor(delta_pose[0, 0]).to(device).float())
        delta_map = moveLocalMap(position, rotation, cart_img, local_map_xy, local_map_res, local_map_zero_idx)
        

        local_map = 0.5*local_map + 0.5*delta_map

        # # show local map with matplotlib
        # plt.clf()
        # plt.imshow(local_map.cpu().numpy(), cmap='gray')
        # plt.title(f"Delta Map at pose {i} (time: {teach_times[i]})")
        # plt.colorbar()
        # # plt.show()
        # plt.draw()
        # plt.pause(0.5)

        # plt.show()
 
    cnt += 1
    # no need to blur
    # local_map = local_map / torch.max(local_map)  # normalize the local map
    # local_map_blurred = torchvision.transforms.functional.gaussian_blur(local_map.unsqueeze(0).unsqueeze(0), 3).squeeze()
    # normalizer = torch.max(local_map) / torch.max(local_map_blurred)
    # local_map_blurred *= normalizer

    # Dump local maps 
    # VTR local maps
     # save one-to-one local map
    if local_map is not None:
            mid_scan_timestamp = teach_vertex_timestamps[index][0]
            cv2.imwrite(local_map_path + str(mid_scan_timestamp) + '.png', local_map.detach().cpu().numpy()*255)
    
    

    if cnt == 4:
        break
    



    # plt.imshow(local_map.cpu().numpy(), cmap='gray')
    # plt.title(f"Local Map at pose {index} (time: {cur_pose_time})")
    # plt.colorbar()
    # plt.show()
