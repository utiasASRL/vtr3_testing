import yaml
import torch
import torchvision
import os
import numpy as np

parent_folder = "/home/leonardo/sam/vtr3_testing"

# for each teach pose, I'm gonna compute the n nearest poses (the distance in some bounds).
# with the relative translormation to the current pose I'm gonna trasform the cartesian point in the current pose
# for each polar image compute the cartesian point, apply the relative trasform, project back

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
config = load_config(os.path.join(parent_folder,'scripts/direct/direct_configs/direct_config_leo.yaml'))
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


max_distance = 2

for index in range(len(teach_times)):
    cur_pose_time = teach_times[index]

    cur_pose = list(teach_vertex_transforms[index][0].values())[0].matrix()

    local_map = torch.zeros((local_map_size, local_map_size)).to(device)

    for i in range(len(teach_times)):
        print(f"Processing pose {i} at time {teach_times[i]}")
        # if i == index:
            # continue
        pose = list(teach_vertex_transforms[i][0].values())[0].matrix()
        # compute the distance between the poses
        print(type(pose))
        delta_pose = np.linalg.inv(pose) @ cur_pose

        if np.linalg.norm(delta_pose[1:3, 3]) > max_distance:
            continue

        azimuths = torch.tensor(teach_azimuth_angles[i]).to(device).float()
        nb_azimuths = torch.tensor(len(azimuths)).to(device)
        polar_image = torch.tensor(teach_polar_imgs[i]).to(device).float()
      
        temp_polar_to_interp = local_map_polar.clone()
        temp_polar_to_interp[:,:,0] -= (azimuths[0])
        temp_polar_to_interp[temp_polar_to_interp[:,:,0]<0] = temp_polar_to_interp[temp_polar_to_interp[:,:,0]<0] + torch.tensor((2*torch.pi, 0)).to(device)
        temp_polar_to_interp[:,:,0] *= ((nb_azimuths) / (2*torch.pi))
        temp_polar_to_interp[:,:,1] -= (radar_res/2.0)
        temp_polar_to_interp[:,:,1] /= radar_res

        cart_img = bilinearInterpolation(polar_image, temp_polar_to_interp)
        
        position = torch.tensor(delta_pose[1:2, 3]).to(device).float()
        rotation = torch.atan2(torch.tensor(delta_pose[1, 0]).to(device).float(), torch.tensor(delta_pose[0, 0]).to(device).float())
        delta_map = moveLocalMap(position, rotation, cart_img, local_map_xy, local_map_res, local_map_zero_idx)

        local_map += delta_map

    # show local map with matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(local_map.cpu().numpy(), cmap='gray')
    plt.title(f"Local Map at pose {index} (time: {cur_pose_time})")
    plt.colorbar()
    plt.show()
