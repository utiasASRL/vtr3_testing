import os
import sys
import yaml
import numpy as np
import cv2
import torch
import torchvision

parent_folder = "/home/sahakhsh/Documents/vtr3_testing"

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

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

config = load_config(os.path.join(parent_folder,'scripts/direct/direct_config_hshmat.yaml'))

result_folder = config.get('output')

# change here
out_path_folder = os.path.join(result_folder,f"grassy_t2_r3/")
if not os.path.exists(out_path_folder):
    os.makedirs(out_path_folder)
    print(f"Folder '{out_path_folder}' created.")
else:
    print(f"Folder '{out_path_folder}' already exists.") 

def localMapToPolarCoord_(cart, radar_res, nb_azimuths):
    with torch.no_grad():
        # Get the polar coordinates of the image
        # Get the new polar coordinates
        polar = torch.zeros((cart.shape[0], cart.shape[1], 2)).to(device)
        polar[:, :, 0] = torch.atan2(cart[:, :, 1, 0], cart[:, :, 0, 0])
        polar[:, :, 1] = torch.sqrt(cart[:, :, 0, 0]**2 + cart[:, :, 1, 0]**2)

        #polar[polar[:,:,0]<0] = polar[polar[:,:,0]<0] + torch.tensor((2*torch.pi, 0)).to(self.device)
        #polar[:,:,0] *= (nb_azimuths / (2*torch.pi))
        #polar[:,:,1] /= radar_res
        return polar

cart_resolution = 0.234
radar_res = 0.040308

alpha = 0.1  # Smoothing factor
opts = load_config(os.path.join(parent_folder,'scripts/direct/warthog_config.yaml'))
local_map_res = float(opts['direct']['local_map_res'])
max_local_map_range = float(opts['direct']['max_local_map_range'])
local_map_size = int(max_local_map_range/local_map_res)*2 + 1
temp_x = (torch.arange( -local_map_size//2, local_map_size//2, 1).to(device) + 1) * local_map_res
X = -temp_x.unsqueeze(0).T.repeat(1,local_map_size)
Y = temp_x.unsqueeze(0).repeat(local_map_size,1)
local_map = torch.zeros((local_map_size, local_map_size)).to(device)
print(f"Shape of local_map is {local_map.shape}, type is {type(local_map)}, and range is {torch.min(local_map)} to {torch.max(local_map)}")
local_map_xy = torch.stack((X, Y), dim=2).unsqueeze(-1).to(device)
local_map_res = torch.tensor(local_map_res).to(device)
local_map_zero_idx = torch.tensor(int(max_local_map_range/local_map_res)).to(device)
local_map_polar = localMapToPolarCoord_(local_map_xy, radar_res, 400)
print(f"Shape of local_map_polar is {local_map_polar.shape}, type is {type(local_map_polar)}, and range is {torch.min(local_map_polar)} to {torch.max(local_map_polar)}")
local_map_mask = (local_map_polar[:,:,1] < max_local_map_range) & (local_map_polar[:,:,1] > float(opts['direct']['min_range']))

local_map_output_folder = os.path.join(out_path_folder, "local_maps")
os.makedirs(local_map_output_folder, exist_ok=True)

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

class RadarFrame:
    def __init__(self, polar, azimuths, timestamps):
        self.polar = polar[:, :].astype(np.float32) / 255.0
        self.azimuths=azimuths
        self.timestamps=timestamps.flatten().astype(np.int64)

def bilinearInterpolation_(im, az_r, with_jac = False):
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

        if not with_jac:
            return img_interp
        else:
            d_I_d_az_r = torch.empty((az_r.shape[0], az_r.shape[1], 1, 2), device=device)
            d_I_d_az_r[:, :, 0, 0] = (Ib - Ia)*local_1_minus_r + (Id - Ic)*local_r
            d_I_d_az_r[:, :, 0, 1] = (Ic - Ia)*local_1_minus_az + (Id - Ib)*local_az
            return img_interp, d_I_d_az_r
    
def cartToLocalMapID_(xy):
    out = torch.empty_like(xy, device=device)
    out[:,:,0,0] = (xy[:,:,0,0] / (-local_map_res)) + local_map_zero_idx
    out[:,:,1,0] = (xy[:,:,1,0] / (local_map_res)) + local_map_zero_idx
    return out
    
# Move localMap
def moveLocalMap_(pos, rot):
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
        new_idx = cartToLocalMapID_(new_xy)

        # Get the new localMap via bilinear interpolation
        local_map = bilinearInterpolation_(local_map, new_idx).squeeze().float()

# create a folder that dumps images from 
for teach_vertex_idx in range(0,teach_times.shape[0],1):
    teach_vertex_time = teach_times[teach_vertex_idx]
    print("The teach vertex time is: ", teach_vertex_time[0])

    polar_image = teach_polar_imgs[teach_vertex_idx]
    azimuth_angles = teach_azimuth_angles[teach_vertex_idx]
    azimuth_timestamps = teach_azimuth_timestamps[teach_vertex_idx]
    print(f"Shape of polar_image is {polar_image.shape}, type is {type(polar_image)}, and range is {np.min(polar_image)} to {np.max(polar_image)}")

    radar_frame = RadarFrame(polar_image, azimuth_angles, azimuth_timestamps)
    # Have to do this preprocessing step
    polar_intensity = torch.tensor(radar_frame.polar).to(device)
    polar_std = torch.std(polar_intensity, dim=1)
    polar_mean = torch.mean(polar_intensity, dim=1)
    polar_intensity -= (polar_mean.unsqueeze(1) + 2*polar_std.unsqueeze(1))
    polar_intensity[polar_intensity < 0] = 0
    polar_intensity = torchvision.transforms.functional.gaussian_blur(polar_intensity.unsqueeze(0), (9,1), 3).squeeze()
    polar_intensity /= torch.max(polar_intensity, dim=1, keepdim=True)[0]
    polar_intensity[torch.isnan(polar_intensity)] = 0
    print(f"Shape of polar_intensity is {polar_intensity.shape}, type is {type(polar_intensity)}, and range is {torch.min(polar_intensity)} to {torch.max(polar_intensity)}")
    
    cart_img = radar_polar_to_cartesian(polar_image, azimuth_angles, radar_res, cart_resolution, 640)
    print(f"Shape of cart_img is {cart_img.shape}, type is {type(cart_img)}, and range is {np.min(cart_img)} to {np.max(cart_img)}")

    # Update local map
    temp_polar_to_interp = local_map_polar.clone()
    azimuths = torch.tensor(azimuth_angles.flatten()).to(device).float()
    nb_azimuths = torch.tensor(len(azimuth_angles.flatten())).to(device) # number of azimuths 400
    temp_polar_to_interp[:,:,0] -= (azimuth_angles[0])
    temp_polar_to_interp[temp_polar_to_interp[:,:,0]<0] = temp_polar_to_interp[temp_polar_to_interp[:,:,0]<0] + torch.tensor((2*torch.pi, 0)).to(device)
    temp_polar_to_interp[:,:,0] *= ((nb_azimuths) / (2*torch.pi))
    temp_polar_to_interp[:,:,1] -= (radar_res/2.0)
    temp_polar_to_interp[:,:,1] /= radar_res
    polar_target = torch.concatenate((polar_target, polar_target[0,:].unsqueeze(0)), dim=0)
    local_map_update = bilinearInterpolation_(polar_target, temp_polar_to_interp, with_jac=False)                        
    step_counter = 1
    if step_counter == 1:
        local_map[local_map_mask] = local_map_update[local_map_mask]
    else:
        moveLocalMap_(frame_pos, frame_rot)
        local_map[local_map_mask] = (1-alpha) * local_map[local_map_mask] + alpha * local_map_update[local_map_mask]























    # Ensure float32 for accumulation
    cart_img = cart_img.astype(np.float32)

    if local_map is None:
        local_map = cart_img.copy()
    else:
        local_map = (1 - alpha) * local_map + alpha * cart_img

    # Save local_map as .png 
    visual_map = np.clip(local_map, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(local_map_output_folder, f"local_map_{teach_vertex_idx:04d}.png"), visual_map)

    

