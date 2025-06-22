import numpy as np
import cv2
# import gp_doppler as gpd



def dopplerUpDown(radar_frame):
    # Remove the last folder from the radar_frame.sensor_root
    base_path = '/'.join(radar_frame.sensor_root.split('/')[:-1])
    doppler_path = base_path + '/doppler_radar/' + radar_frame.frame + '.png'
    doppler_img = cv2.imread(doppler_path, cv2.IMREAD_GRAYSCALE)
    up_chrips = doppler_img[:,10]
    return up_chrips

def checkChirp(radar_frame):
    up_chirps = dopplerUpDown(radar_frame)
    # Check that all the even chirps are up and all the odd chirps are down
    if not np.all(up_chirps[::2] == 255) or not np.all(up_chirps[1::2] == 0):
        chirp_up = True
    else:
        chirp_up = False
    return chirp_up


def wrap_angle(theta):
    if(theta > np.pi):
        return theta - 2*np.pi
    elif(theta < -np.pi):
        return theta + 2*np.pi
    else:
        return theta


def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix SO(3) to Euler angles (roll, pitch, yaw).

    Parameters:
        R (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
        tuple: A tuple containing roll, pitch, and yaw angles (in radians).
    """
    assert R.shape == (3, 3), "Input rotation matrix must be 3x3."

    # Check if the matrix is a valid rotation matrix
    if not np.allclose(np.dot(R.T, R), np.eye(3)) or not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError("Input matrix is not a valid rotation matrix.")

    # Extract the Euler angles
    pitch = -np.arcsin(R[2, 0])

    if np.isclose(np.cos(pitch), 0):
        # Gimbal lock case
        roll = 0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return roll, pitch, yaw


def one_image_to_one_image(teach_frame,repeat_frame, config_warthog, radar_resolution):
    
    gp_state_estimator = gpd.GPStateEstimator(config_warthog, radar_resolution)
    state = gp_state_estimator.pairwiseRegistration(teach_frame, repeat_frame)

    return state

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


import torch
import torchvision.transforms.functional
def preprocessing_polar_image(polar_image, device):
    polar_intensity = torch.tensor(polar_image).to(device)
    polar_std = torch.std(polar_intensity, dim=1)
    polar_mean = torch.mean(polar_intensity, dim=1)
    polar_intensity -= (polar_mean.unsqueeze(1) + 2*polar_std.unsqueeze(1))
    polar_intensity[polar_intensity < 0] = 0
    polar_intensity = torchvision.transforms.functional.gaussian_blur(polar_intensity.unsqueeze(0), (9,1), 3).squeeze()
    polar_intensity /= torch.max(polar_intensity, dim=1, keepdim=True)[0]
    polar_intensity[torch.isnan(polar_intensity)] = 0

    return polar_intensity


    