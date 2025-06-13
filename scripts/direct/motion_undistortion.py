## this file aims to undistort the radar images using the current odometry estimates
## as well as all the other preprocessing steps

import numpy as np
import torch
import torchvision


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



def motion_undistortion(polar_images, odometry, device):
    """
    Undistort the radar image using the current odometry estimates.
    """