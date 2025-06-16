import yaml
import torch
import torchvision
import os
import numpy as np

from pylgmath import Transformation
import cv2
import pyboreas as pb

import matplotlib.pyplot as plt
# plt.ion()  # Turn on interactive mode

import sys
parent_folder = "/home/samqiao/ASRL/vtr3_testing"
if parent_folder not in sys.path:
    sys.path.append(parent_folder)

T_radar_robot =  Transformation(T_ba = np.array([[1.000, 0.000, 0.000, 0.025],
                                                 [0.000, -1.000 , 0.000, -0.002],
                                                 [0.000 ,0.000, -1.000 , 1.032],
                                                 [0.000 , 0.000 ,0.000, 1.000]])) # note the radar frame is upside down


# Leo's comments: for each teach pose, I'm gonna compute the n nearest poses (the distance in some bounds).
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
    # look at last commit the function is change ..... from cedric
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

        # Get the coordinate of the new localMap in the former localMap # TODO using the motion model
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


max_range_idx_direct = torch.tensor(int(np.floor(config_warthog['direct']['max_range'] / radar_res)))
min_range_idx_direct = torch.tensor(int(np.ceil(config_warthog['direct']['min_range'] / radar_res)))

print("max_range_idx_direct: ", max_range_idx_direct)
print("min_range_idx_direct: ", min_range_idx_direct)

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
out_path_folder = os.path.join(result_folder,f"mars_t1_r2/")
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
# 7. teach vertex ids
teach_vertex_ids = teach_df['teach_vertex_ids']


max_distance = 3 # 2 m 
max_time_dist = 4 # 4 secs

# save path for local maps
local_map_path = "/home/samqiao/ASRL/vtr3_testing/scripts/direct/mars_t1_r2"
local_map_path = local_map_path + '/local_map_vtr/'
local_map_blurred_path = local_map_path + '_blurred/'
os.makedirs(local_map_path, exist_ok=True)

cnt = 0 
first_pose = None

for index in range(len(teach_times)): # for every pose
    print(f"--------------------processing vertex index: {index}---------------------  percentage processed: {(index / len(teach_times)) * 100:.4f}")

    cur_pose_time = teach_times[index]

    cur_pose = teach_vertex_transforms[index][0][teach_times[index][0]] # T_robot_world

    # if (first_pose is None):
    #     first_pose = cur_pose

    local_map = torch.zeros((local_map_size, local_map_size)).to(device)

    deltas = {}

    poses = []

    for i in range(len(teach_times)): # for the neighbouring poses
        id = teach_vertex_ids[i][0]

        if id in deltas.keys():
            # print("continuing cause id is already seen")
            continue

        # print(teach_times[i][0])
        # print(id)
        # print(deltas.keys())


        # print("sam:", teach_times[i][0], cur_pose_time[0])
        neighbour_pose = teach_vertex_transforms[i][0][teach_times[i][0]] # T_robot_world  repeat_edge_transforms[repeat_vertex_idx][0][repeat_vertex_time[0]]   
        

        # compute the distance between the poses
        # print(type(pose))
        delta_pose =  cur_pose @ neighbour_pose.inverse() # T_radar_robot @ delta_pose @ T_radar_robot.inverse().matrix() # need to do a transofrm # check here

        delta_pose = T_radar_robot @ delta_pose @ T_radar_robot.inverse()  # T_radar_robot @ delta_pose @ T_radar_robot.inverse().matrix

        delta_time = np.abs(teach_times[i][0] - cur_pose_time[0])
        if np.linalg.norm(delta_pose.matrix()[0:2, 3]) > max_distance or delta_time > max_time_dist:
            # print("skipping cause the vertex are too far")
            continue


        poses.append(T_radar_robot @ neighbour_pose) # only takes the poses we care about pls
        deltas[id] = delta_pose
        # print("id: ", id)
        # print("delta: ", delta_pose)

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

        # sam: crop range
        # polar_intensity = polar_intensity[min_range_idx_direct:, : max_range_idx_direct] # crop the polar image to the range we care about


        # if i == index:
        #     print("haha")
        #     cart_resolution = 0.234
        #     cart_pixel_width = 640
        #     cart_polar_intensity = pb.utils.radar.radar        # # # show local map with matplotlib
        # plt.clf()
        # plt.imshow(local_map.cpu().numpy(), cmap='gray')
        # plt.title(f"Delta Map at pose {i} (time: {teach_times[i]})")
        # plt.colorbar()
        # # plt.show()
        # plt.draw()
        # plt.pause(0.5)

        # plt.show()

        temp_polar_to_interp = local_map_polar.clone()            
        temp_polar_to_interp[:,:,0] -= (azimuths[0])
        temp_polar_to_interp[temp_polar_to_interp[:,:,0]<0] = temp_polar_to_interp[temp_polar_to_interp[:,:,0]<0] + torch.tensor((2*torch.pi, 0)).to(device)
        temp_polar_to_interp[:,:,0] *= ((nb_azimuths) / (2*torch.pi))
        temp_polar_to_interp[:,:,1] -= (radar_res/2.0)
        temp_polar_to_interp[:,:,1] /= radar_res

        cart_img = bilinearInterpolation(polar_intensity, temp_polar_to_interp)
        
        position = torch.tensor(delta_pose.inverse().matrix()[0:2, 3]).to(device).float()
        rotation = torch.atan2(torch.tensor(delta_pose.inverse().matrix()[1, 0]).to(device).float(), torch.tensor(delta_pose.inverse().matrix()[0, 0]).to(device).float()) # did an inverse here haha seems to work
        # print("sam: rotation: ", rotation)
        # we need to wrap the rotation to [-pi, pi]
        # if rotation > np.pi:
        #     rotation -= 2 * np.pi
        # elif rotation < -np.pi:
        #     rotation += 2 * np.pi
        delta_map = moveLocalMap(position, rotation, cart_img, local_map_xy, local_map_res, local_map_zero_idx)

        # motion undistortion: local_xy should contain the cartesian coordinates of the local map
        # I need to apply the motion undistortion to the local_map_xy using the velocity
        

        local_map += delta_map

        # # # show local map with matplotlib
        # plt.clf()
        # plt.imshow(local_map.cpu().numpy(), cmap='gray')
        # plt.title(f"Delta Map at pose {i} (time: {teach_times[i]})")
        # plt.colorbar()
        # # plt.show()
        # plt.draw()
        # plt.pause(0.5)

        # plt.show()

    # print("numer of deltas: ", len(deltas))
    # print("numer of poses: ", len(poses))

    # # sort deltas by the key
    deltas = dict(sorted(deltas.items(), key=lambda item: item[0]))

    first_pose_t = teach_vertex_transforms[0][0][teach_times[0][0]] # T_robot_world
    # print("sam first pose: ", first_pose_t.matrix())

    # current pose is wrt the world frame
    # delta pose takes the neighbour and express in the current frame

    traj = []

    for key, delta_pose in deltas.items():

        neighbour_pose =  (delta_pose.inverse() @ T_radar_robot @ cur_pose).inverse()    # delta_pose = cur_pose @ neighbour_pose.inverse() # T_radar_robot @ delta_pose @ T_radar_robot.inverse().matrix() # need to do a transofrm # check here

        position = neighbour_pose.r_ab_inb() # neighbour pose x,y, and z

        traj.append(position)

    import matplotlib.pyplot as plt

    x_traj = [p[0].item() for p in traj]
    y_traj = [p[1].item() for p in traj]
    x_poses = [p.r_ba_ina()[0].item() for p in poses]
    y_poses = [p.r_ba_ina()[1].item() for p in poses]

    # plt.figure()
    # plt.plot(x_traj, y_traj, 'b.-', label='Trajectory')
    # plt.plot(x_poses, y_poses, 'r.', label='Poses')
    # plt.legend()
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.show()
 
    cnt += 1
    

    local_map = local_map / torch.max(local_map) # normalize the local map
    
    # blur the map and normalize it
    local_map_blurred = torchvision.transforms.functional.gaussian_blur(local_map.unsqueeze(0).unsqueeze(0), 3).squeeze()
    normalizer = torch.max(local_map) / torch.max(local_map_blurred)
    local_map_blurred *= normalizer

    # Dump local maps 
    # VTR local maps
     # save one-to-one local map
    # break
    if local_map is not None:
            mid_scan_timestamp = teach_vertex_timestamps[index][0]
    
            cv2.imwrite(local_map_path + str(mid_scan_timestamp) + '.png', local_map.detach().cpu().numpy()*255)
            
            if local_map_blurred is not None:
            # save blurred local map
                cv2.imwrite(local_map_blurred_path + str(mid_scan_timestamp) + '.png', local_map_blurred.detach().cpu().numpy()*255)


    

    # plt.imshow(local_map.cpu().numpy(), cmap='gray')
    # plt.title(f"Local Map at pose {index} (time: {cur_pose_time})")
    # plt.colorbar()
    # plt.show()
