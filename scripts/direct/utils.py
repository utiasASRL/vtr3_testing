import numpy as np
import cv2
import gp_doppler as gpd



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
    