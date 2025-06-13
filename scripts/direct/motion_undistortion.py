## this file aims to undistort the radar images using the current odometry estimates
## as well as all the other preprocessing steps

import numpy as np
import torch
import torchvision

from utils import * # import radar_polar_to_cartesian

def preprocessing_polar_image(polar_image, device):
    
    # add normalization to polar_image
    polar_intensity = torch.tensor(polar_image).to(device)
    polar_std = torch.std(polar_intensity, dim=1)
    polar_mean = torch.mean(polar_intensity, dim=1)
    polar_intensity -= (polar_mean.unsqueeze(1) + 2*polar_std.unsqueeze(1))
    polar_intensity[polar_intensity < 0] = 0
    polar_intensity = torchvision.transforms.functional.gaussian_blur(polar_intensity.unsqueeze(0), (9,1), 3).squeeze()
    polar_intensity /= torch.max(polar_intensity, dim=1, keepdim=True)[0]
    polar_intensity[torch.isnan(polar_intensity)] = 0


    return polar_intensity



def motion_undistortion(polar_image, azimuth_angles, azimuth_timestamps, T_v_w, dt, device):
    """
    Undistort the radar image using the current odometry estimates.
    """
    polar_intensity = preprocessing_polar_image(polar_image, device)
    # get the velocity in polar coordinates

    azimuths = torch.tensor(azimuth_angles).to(device).float()
    nb_azimuths = torch.tensor(len(azimuths)).to(device)

    # you need to get the cartesian image first and then undistort it

    velocity_estimated = torch.zeros_like(azimuths) ### assume constant velocity



    undistorted_polar_intensity = torch.zeros_like(polar_intensity)

    return undistorted_polar_intensity