import yaml
import gp_doppler as gpd
import pyboreas as pb
# import warthog_data_loader as wd
# import mulran_data_loader as md
import time
import numpy as np
import pandas as pd
import os
import utils
import cv2


def main():
    # Load the configuration file
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # if config['data']['multi_sequence']:
    #     if config['data']['type'] == 'boreas':
    #         db = pb.BoreasDataset(config['data']['data_path'])
    #         sequences = db.sequences
    #     elif config['data']['type'] == 'warthog':
    #         db = wd.WarthogDataset(config['data']['data_path'])
    #         sequences = db.sequences
    #     elif config['data']['type'] == 'mulran':
    #         db = md.MulranDataset(config['data']['data_path'])
    #         sequences = db.sequences
    #     else:
    #         raise ("Unknown data type")
    # else:
    #     # Get the last repo of the data_path
    #     sequence_id = config['data']['data_path'].split('/')[-1]
    #     # Get the base path
    #     base_path = '/'.join(config['data']['data_path'].split('/')[:-1])
    #     if config['data']['type'] == 'boreas':
    #         db = pb.BoreasDataset(base_path)
    #     elif config['data']['type'] == 'warthog':
    #         db = wd.WarthogDataset(base_path)
    #     elif config['data']['type'] == 'mulran':
    #         db = md.MulranDataset(base_path)
    #     else:
    #         raise ("Unknown data type")
    #     sequences = []
    #     sequences.append(db.get_seq_from_ID(sequence_id))
        

    # Check if the output path exists
    os.makedirs('output', exist_ok=True)

    # Check the visualisation and saving options
    visualise = config['log']['display']
    save_images = config['log']['save_images']
    doppler_radar = config['radar']['doppler_enabled']
    if not doppler_radar:
        chirp_up = config['radar']['chirp_up']
    potential_chirp_flip = False
    use_gyro = 'gyro' in config['estimation']['motion_model']
    verbose = config['log']['verbose']

    # Parameters for bias estimation
    gyro_bias_alpha = 0.01
    estimate_gyro_bias = False

    # Check if the data is in boreas format
    boreas_format = config['data']['type'] == 'boreas'
    mulran_format = config['data']['type'] == 'mulran'

    # Prepare for the vy bias estimation
    if doppler_radar:
        estimate_vy_bias = config['estimation']['estimate_doppler_vy_bias']
        if estimate_vy_bias:
            T_axle_radar = np.array(config['estimation']['T_axle_radar'])
            if 'vy_bias_alpha' in config['estimation']:
                vy_bias_alpha = config['estimation']['vy_bias_alpha']
            else:
                vy_bias_alpha = 0.01
            vy_bias = 0.0
    else:
        estimate_vy_bias = False
    

    # Check the mode
    for seq in sequences:

        if estimate_vy_bias:
            vy_bias_log = []
        # Create the GP model
        temp_radar_frame = seq.get_radar(0)
        res = temp_radar_frame.resolution
        if mulran_format:
            res*=1.00#15
        gp_state_estimator = gpd.GPStateEstimator(config, res)
        temp_radar_frame.unload_data()


        if use_gyro:
            gyro_bias = 0.0
            gyro_bias_log = []
            gyro_bias_counter = 0
            gyro_bias_initialised = False
            previous_vel_null = False
            estimate_gyro_bias = config['estimation']['gyro_bias_estimation']

            if boreas_format:
                if config['imu']['type'] == 'applanix':
                    imu_path = os.path.join(seq.seq_root, 'applanix', 'imu_raw.csv')
                    imu_data = np.loadtxt(imu_path, delimiter=',', skiprows=1)
                    imu_time = imu_data[:, 0]
                    imu_gyro = np.stack((imu_data[:, 3], imu_data[:, 2], imu_data[:, 1]), axis=1)
                    T_applanix_radar = seq.calib.T_applanix_lidar @ np.linalg.inv(seq.calib.T_radar_lidar)
                    imu_gyro = imu_gyro @ T_applanix_radar[:3, :3]
                    imu_yaw = -imu_gyro[:, 2]
                elif config['imu']['type'] == 'dmu':
                    imu_path = os.path.join(seq.seq_root, 'applanix', 'dmu_imu.csv')
                    imu_data = np.loadtxt(imu_path, delimiter=',', skiprows=1)
                    imu_time = imu_data[:, 0] * 1e-9
                    imu_yaw = imu_data[:, 9]
                else:
                    print("Unknown IMU type")
                    return
            else:
                imu_time, imu_gyro = seq.get_imu_data()
                T_radar_osimu = seq.calib['imu_in_radar']
                imu_gyro = imu_gyro @ T_radar_osimu[:3, :3]
                imu_yaw = imu_gyro[:, 2]
                if mulran_format:
                    imu_time = imu_time - 0.05


            gp_state_estimator.motion_model.setGyroData(imu_time, imu_yaw)
            min_gyro_sample_bias = config['imu']['min_time_bias_init'] / np.mean(np.diff(imu_time))
            if estimate_vy_bias:
                vy_bias = config['estimation']['vy_bias_prior']


            if 'gyro_bias_prior' in config['estimation']:
                gyro_bias = config['estimation']['gyro_bias_prior']
                gyro_bias_initialised = True
                gyro_bias_counter = min_gyro_sample_bias + 1

        degraded_log = []



        # Logging when using constant w model
        estimate_ang_vel = 'const_w' in config['estimation']['motion_model']
        if estimate_ang_vel:
            yaw_list_gt = []
            yaw_list_est = []


        # Prepare output folders
        seq_output_path = 'output/' + seq.ID
        os.makedirs(seq_output_path, exist_ok=True)
        odom_output_path = seq_output_path + '/odometry_result'
        if os.path.exists(odom_output_path):
            os.system('rm -r ' + odom_output_path)
        os.makedirs(odom_output_path)
        odom_output_path = odom_output_path + '/' + seq.ID + '.txt'
        other_log_path = seq_output_path + '/other_log'
        os.makedirs(other_log_path, exist_ok=True)
        if save_images:
            image_output_path = seq_output_path + '/images'
            if os.path.exists(image_output_path):
                os.system('rm -r ' + image_output_path)
            os.makedirs(image_output_path, exist_ok=True)



        # Variables to log the time
        time_sum = 0
        opti_time_sum = 0
        time_counter = 0
        # Variables to log the error
        if boreas_format:
            error_norm = []
            gt_norm = []
            gt_vel = []
            est_vel = []
        computation_time = []
        # Tracking of the chirp up for doppler radar
        if doppler_radar:
            prev_chirp_up = None
        else:
            chirp_up = not config['radar']['chirp_up']

        
        # Loop over the radar frames
        end_id = len(seq.radar_frames)
        start_id = 0
        for i in range(start_id, end_id):
            time_start = time.time()

            # Load the radar frame
            radar_frame = seq.get_radar(i)
            
            if use_gyro:
                if gyro_bias_initialised:
                    gyro_bias_log.append(gyro_bias)
                else:
                    gyro_bias_log.append(0)

            # Update the gyro bias if needed
            if use_gyro and estimate_gyro_bias and gyro_bias_initialised:
                gp_state_estimator.motion_model.setGyroBias(gyro_bias)

            
            # Check the chirp up/down status to account for the
            # hardware problem of the doppler radar
            if doppler_radar:
                potential_chirp_flip = False
                chirp_up = utils.checkChirp(radar_frame)
                if i != end_id-1:
                    next_radar_frame = seq.get_radar(i+1)
                    next_chirp_up = utils.checkChirp(next_radar_frame)
                    next_radar_frame.unload_data()
                    if next_chirp_up != chirp_up:
                        potential_chirp_flip = True
                if prev_chirp_up is not None and prev_chirp_up != chirp_up:
                    potential_chirp_flip = True
                prev_chirp_up = chirp_up
            

            # Display of the progress
            if time_counter == 0:
                print("Frame " + str(i-start_id+1) + " / " + str(end_id-start_id), end='\r')
            else:
                print("Frame " + str(i-start_id+1) + " / " + str(end_id-start_id) + " - Avg. opti: " + str(round(opti_time_sum/time_counter,3)) + "s, time left (including visualisation): " + str(round((end_id-i)*time_sum/time_counter/60, 3)) + "min    ", end='\r')

            #if mulran_format:
                #radar_frame.azimuths = radar_frame.azimuths - 0.003
                #radar_frame.timestamps = radar_frame.timestamps + 25000
                #radar_frame.timestamp = radar_frame.timestamp + 0.025



            # Optimise the scan velocity
            time_start = time.time()
            polar_img = radar_frame.polar
            state = gp_state_estimator.odometryStep(polar_img, radar_frame.azimuths.flatten(), radar_frame.timestamps.flatten(), chirp_up, potential_chirp_flip)

            degraded_log.append(gp_state_estimator.degraded_log)
            
            # Get the velocity 
            velocity = state[:2]
            if config['estimation']['motion_model'] == 'const_body_acc_gyro':
                velocity = state[:2] * (1+state[2]*0.125)
            if estimate_vy_bias and np.linalg.norm(velocity) > 3:
                gp_state_estimator.vy_bias = 0.0
                doppler_vel = gp_state_estimator.getDopplerVelocity()
                doppler_vel = np.concatenate([doppler_vel, [0]])
                print("\nDoppler velocity: ", doppler_vel)
                if use_gyro:
                    # Get the average angular velocity between the first and last azimuth
                    gyro_idx = np.logical_and(imu_time >= radar_frame.timestamps[0]*1e-6, imu_time <= radar_frame.timestamps[-1]*1e-6)
                    gyro_data = imu_yaw[gyro_idx]
                    if len(gyro_data) == 0:
                        gyro_data = np.array([0, 0, 0])
                    else:
                        gyro_data = np.mean(gyro_data)
                        gyro_data = T_axle_radar[:3, :3] @ np.array([0, 0, gyro_data])
                    axle_vel = T_axle_radar[:3, :3] @ doppler_vel + np.cross(gyro_data, T_axle_radar[3, :3])
                else:
                    axle_vel = T_axle_radar[:3, :3] @ doppler_vel
                vy = (T_axle_radar[:3,:3].T@(np.array([0, axle_vel[1], 0])))[1]
                vy_bias = vy_bias_alpha * vy + (1-vy_bias_alpha) * vy_bias
                gp_state_estimator.vy_bias = vy_bias
            if estimate_vy_bias:
                vy_bias_log.append(vy_bias)
                print("\nVy bias: ", vy_bias)

            time_end = time.time()


            # Time the optimisation (remove the first few warmup iterations)
            if time_counter == 5:
                opti_time_sum = (time_end - time_start)*5
            opti_time_sum += time_end - time_start
            computation_time.append(time_end - time_start)

            # Log the velocity
            if boreas_format:
                error_norm.append(np.linalg.norm(velocity) - np.linalg.norm(radar_frame.body_rate[:3]))
                gt_norm.append(np.linalg.norm(radar_frame.body_rate[:3]))
                gt_vel.append(radar_frame.body_rate[:3])
                est_vel.append(velocity)

            # Display information
            if verbose:
                print("\n")
                if config['estimation']['motion_model'] == 'const_body_acc_gyro':
                    print("Acceleration: ", state[2])
                print("Velocity: ", velocity)
                if boreas_format:
                    print("Velocity GT: ", radar_frame.body_rate[:2].flatten())
                    print("Diff norm: ", np.linalg.norm(velocity) - np.linalg.norm(radar_frame.body_rate[:3]))
                    print("RMSE so far", np.sqrt(np.mean(np.array(error_norm)**2)))
                    print("Mean error so far", np.mean(error_norm))
                print("\n")

            # Log the angular velocity if estimating the angular velocity
            if estimate_ang_vel and boreas_format:
                yaw_list_gt.append(radar_frame.body_rate[5])
                yaw_list_est.append(state[-1])



            # Estimate the gyro bias when the velocity is null
            if estimate_gyro_bias and np.linalg.norm(velocity) < 0.05:
                if previous_vel_null:
                    # Get gyro measurements between the first and last azimuth
                    gyro_idx = np.logical_and(imu_time >= radar_frame.timestamps[0]*1e-6, imu_time <= radar_frame.timestamps[-1]*1e-6)
                    gyro_data = imu_yaw[gyro_idx]
                    invalid = False
                    if gyro_bias_counter != 0 and (np.abs(np.mean(gyro_data)-gyro_bias) > 2*np.abs(gyro_bias)):
                        invalid = True
                    if not invalid:
                        if not gyro_bias_initialised:
                            gyro_bias += np.sum(gyro_data)
                            gyro_bias_counter += len(gyro_data)
                            if gyro_bias_counter > min_gyro_sample_bias:
                                gyro_bias /= gyro_bias_counter
                                gyro_bias_initialised = True
                        else:
                            gyro_bias = gyro_bias_alpha * np.mean(gyro_data) + (1-gyro_bias_alpha) * gyro_bias
                previous_vel_null = True
            if estimate_gyro_bias and verbose:
                if gyro_bias_initialised:
                    print("Gyro bias: ", gyro_bias)
                else:
                    print("Gyro bias not initialised")





            # Get the position and rotation for the evaluation
            current_pos, current_rot = gp_state_estimator.getAzPosRot()
            if current_pos is not None:
                current_pos = current_pos.squeeze()
                current_rot = current_rot.squeeze()
                
                # Get the id closest to the GT
                mid_id = np.argmin(np.abs(radar_frame.timestamps.flatten().astype(np.float64)*1e-6 - radar_frame.timestamp))

                trans_mat = np.array([[np.cos(current_rot[mid_id]), -np.sin(current_rot[mid_id]), 0, current_pos[mid_id][0]],
                                    [np.sin(current_rot[mid_id]), np.cos(current_rot[mid_id]), 0, current_pos[mid_id][1]],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

                if boreas_format or mulran_format:
                    trans_mat = np.linalg.inv(trans_mat)
                else:
                    R = np.array([[0, -1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
                    trans_mat = R @ trans_mat

                data = np.concatenate([radar_frame.timestamps[mid_id], trans_mat[:3, :].flatten()]).reshape(1, -1)
                df_data = pd.DataFrame(data)
                df_data[0] = df_data[0].astype(int)
                if not os.path.exists(odom_output_path):
                    df_data.to_csv(odom_output_path, header=None, index=None, sep=' ')
                else:
                    df_data.to_csv(odom_output_path, mode='a', header=None, index=None, sep=' ')


            if visualise or save_images:
                img = gp_state_estimator.generateVisualisation(radar_frame, 500, radar_frame.resolution*radar_frame.polar.shape[1]/(250*np.sqrt(2)), inverted=True, text=False)
            if visualise:
                cv2.imshow('Image', img)
                cv2.waitKey(5)


            # Save the image (with 6 digits)
            if save_images:
                cv2.imwrite(image_output_path + '/frame_' + str(i-start_id).zfill(6) + '.png', img)


            radar_frame.unload_data()

            # Time the loop for statistics
            time_end = time.time()
            if time_counter == 1:
                time_sum = time_end - time_start
            time_sum += time_end - time_start
            time_counter += 1


        # Save the other logs
        if boreas_format:
            gt_vel_path = other_log_path + '/gt_vel.txt'
            gt_vel = np.array(gt_vel).squeeze()
            df_gt_vel = pd.DataFrame(gt_vel)
            df_gt_vel.to_csv(gt_vel_path, header=None, index=None, sep=' ')
            est_vel_path = other_log_path + '/est_vel.txt'
            est_vel = np.array(est_vel).squeeze()
            df_est_vel = pd.DataFrame(est_vel)
            df_est_vel.to_csv(est_vel_path, header=None, index=None, sep=' ')

            # Save the angular velocity
            if estimate_ang_vel:
                yaw_list_gt = np.array(yaw_list_gt)
                yaw_list_est = np.array(yaw_list_est)
                yaw_list_path = other_log_path + '/est_ang_vel.txt'
                df_yaw_list = pd.DataFrame(yaw_list_est)
                df_yaw_list.to_csv(yaw_list_path, header=None, index=None, sep=' ')
                yaw_list_path = other_log_path + '/gt_ang_vel.txt'
                df_yaw_list = pd.DataFrame(yaw_list_gt)
                df_yaw_list.to_csv(yaw_list_path, header=None, index=None, sep=' ')
        
        # Save the computation time
        computation_time_path = other_log_path + '/computation_time.txt'
        computation_time = np.array(computation_time)
        df_computation_time = pd.DataFrame(computation_time)
        df_computation_time.to_csv(computation_time_path, header=None, index=None, sep=' ')

        if use_gyro:
            gyro_bias_path = other_log_path + '/gyro_bias.txt'
            gyro_bias_log = np.array(gyro_bias_log)
            df_gyro_bias = pd.DataFrame(gyro_bias_log)
            df_gyro_bias.to_csv(gyro_bias_path, header=None, index=None, sep=' ')
        
        if estimate_vy_bias:
            vy_bias_path = other_log_path + '/vy_bias.txt'
            vy_bias_log = np.array(vy_bias_log)
            df_vy_bias = pd.DataFrame(vy_bias_log)
            df_vy_bias.to_csv(vy_bias_path, header=None, index=None, sep=' ')

        degraded_log_path = other_log_path + '/degraded.txt'
        degraded_log = np.array(degraded_log)
        df_degraded = pd.DataFrame(degraded_log)
        df_degraded.to_csv(degraded_log_path, header=None, index=None, sep=' ')


    if visualise:
        cv2.destroyAllWindows()
    print("")


if __name__ == '__main__':
    main()