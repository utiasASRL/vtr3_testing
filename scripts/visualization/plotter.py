from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
import numpy as np
import os

# import pandas as pd
from pylgmath import Transformation
# import pylgmath


import sys
parent_folder = "/home/leonardo/vtr3_testing"

import yaml

# Insert path at index 0 so it's searched first
sys.path.insert(0, parent_folder)

from deps.path_tracking_error.fcns import *


print("Current working dir", os.getcwd())

T_novatel_robot =  Transformation(T_ba = np.array([[1.000, 0.000, 0.000, 0.550],
  [0.000, 1.000 , 0.000, 0.000],
  [0.000 ,0.000, 1.000 , -1.057],
  [0.000 , 0.000 ,0.000, 1.000]]))

T_radar_robot =  Transformation(T_ba = np.array([[1.000, 0.000, 0.000, 0.025],
                                                 [0.000, -1.000 , 0.000, -0.002],
                                                 [0.000 ,0.000, -1.000 , 1.032],
                                                 [0.000 , 0.000 ,0.000, 1.000]]))



def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file.

    :param config_path: Path to the YAML configuration file.
    :return: A dictionary representing the configuration.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


# need to write a plotter class to plot the data
class Plotter:
    """
    Plotter class to plot the
    """
    def __init__(self):
        # maybe here can be the plotting parameters in the 

        return None

    
    def set_data(self,path_to_data):
        
        # load the data
        TEACH_FOLDER = os.path.join(path_to_data,"teach")
        REPEAT_FOLDER = os.path.join(path_to_data, f"repeat")
        RESULT_FOLDER = os.path.join(path_to_data, f"direct")

        if not os.path.exists(TEACH_FOLDER):
            raise FileNotFoundError(f"Teach folder {TEACH_FOLDER} does not exist.")
        if not os.path.exists(REPEAT_FOLDER):
            raise FileNotFoundError(f"Repeat folder {REPEAT_FOLDER} does not exist.")
        if not os.path.exists(RESULT_FOLDER):
            raise FileNotFoundError(f"Result folder {RESULT_FOLDER} does not exist.")

        teach_df = np.load(os.path.join(TEACH_FOLDER, "teach.npz"),allow_pickle=True)
        # in the teach
        # 1. (932,400,1712) images
        self.teach_polar_imgs = teach_df['teach_polar_imgs']
        # 2. (932,400, 1) azimuth angles
        self.teach_azimuth_angles = teach_df['teach_azimuth_angles']
        # 3. (932,400, 1) azimuth timestamps
        self.teach_azimuth_timestamps = teach_df['teach_azimuth_timestamps']
        # 4. (932,1) vertex timestamps
        self.teach_vertex_timestamps = teach_df['teach_vertex_timestamps']
        # 5. Pose at each vertex: (932,4,4)
        self.teach_vertex_transforms = teach_df['teach_vertex_transforms']
        # 6. teach vertext time
        self.teach_times = teach_df['teach_times']


        # load the repeat data
        repeat_df = np.load(os.path.join(REPEAT_FOLDER, f"repeat.npz"),allow_pickle=True)
        # in the repeat
        self.repeat_times = repeat_df['repeat_times']
        self.repeat_polar_imgs = repeat_df['repeat_polar_imgs']
        self.repeat_azimuth_angles = repeat_df['repeat_azimuth_angles']
        self.repeat_azimuth_timestamps = repeat_df['repeat_azimuth_timestamps']
        self.repeat_vertex_timestamps = repeat_df['repeat_vertex_timestamps']
        self.repeat_edge_transforms = repeat_df['repeat_edge_transforms']
        self.vtr_estimated_ptr = repeat_df['dist']


        # load the result data
        result_df = np.load(os.path.join(RESULT_FOLDER, f"result.npz"),allow_pickle=True)
        self.vtr_norm = result_df['vtr_norm']
        self.gps_norm = result_df['gps_norm']
        self.dir_norm = result_df['dir_norm']
        self.direct_se2_pose = result_df['direct_se2_pose']
        self.vtr_se2_pose = result_df['vtr_se2_pose']
        self.gps_teach_pose = result_df['gps_teach_pose']
        self.gps_repeat_pose = result_df['gps_repeat_pose']

        # load dro results
        dro_df = np.load(os.path.join(RESULT_FOLDER, f"dro_result.npz"),allow_pickle=True)
        # make sure the dro result is there
        if not dro_df:
            raise FileNotFoundError(f"DRO result folder {RESULT_FOLDER} does not exist.")
        
        # in the dro
        self.r_teach_world_dro = dro_df['r_teach_world_dro'] # this is actually redundant here
        self.r_repeat_world_dro = dro_df['r_repeat_world_dro']
        self.dro_se2_pose = dro_df['dro_se2_pose']

        # teach_ppk
        teach_ppk_df = np.load(os.path.join(TEACH_FOLDER, "teach_ppk.npz"),allow_pickle=True)
        if not teach_ppk_df:
            raise FileNotFoundError(f"Teach PPK folder {TEACH_FOLDER} does not exist.")
        # in the teach
        self.r2_pose_teach_ppk = teach_ppk_df['r2_pose_teach_ppk'] # this is actually redundant here

        # repeat_ppk
        repeat_ppk_df = np.load(os.path.join(REPEAT_FOLDER, "repeat_ppk.npz"),allow_pickle=True)
        if not repeat_ppk_df:
            raise FileNotFoundError(f"Repeat PPK folder {REPEAT_FOLDER} does not exist.")
        # in the repeat
        self.r2_pose_repeat_ppk = repeat_ppk_df['r2_pose_repeat_ppk']

        # gps ptr
        self.gps_ptr = self.get_gps_path_tracking_error()
        print("gps_ptr shape:", self.gps_ptr.shape)

        # direct ptr
        self.dir_ptr = self.get_direct_path_tracking_error()
        print("dir_ptr shape:", self.dir_ptr.shape)


        # dor ptr 
        self.dro_ptr = self.get_dro_path_tracking_error()
        print("dro_ptr shape:", self.dro_ptr.shape)


    def plot(self):

        print("vtr_se2_pose shape:", self.vtr_se2_pose.shape)
        print("direct_se2_pose shape:", self.direct_se2_pose.shape)

        plt.figure(1) # need to make it 3 by 1
        plt.subplot(3, 1, 1)
        plt.plot(self.repeat_times, self.vtr_se2_pose[:,0], label='VTR x')
        plt.plot(self.repeat_times, self.direct_se2_pose[:,0],label = 'Direct x')
        plt.title('VTR Direct se2 pose in x,y, theta')
        plt.ylabel('x (m)')
        plt.grid()
        plt.subplot(3, 1, 2)
        plt.plot(self.repeat_times, self.vtr_se2_pose[:,1], label='VTR y')
        plt.plot(self.repeat_times, self.direct_se2_pose[:,1],label = 'Direct y')
        plt.ylabel('y (m)')
        plt.grid()
        plt.subplot(3, 1, 3)
        plt.plot(self.repeat_times, self.vtr_se2_pose[:,2], label='VTR')
        plt.plot(self.repeat_times, self.direct_se2_pose[:,2],label = 'Direct')
        plt.ylabel('theta (rad)')
        plt.grid()
        plt.xlabel('Repeat Times')
        plt.legend()

        plt.show()


    def plot_path_tracking_error(self):
        print("--------In function plot_path_tracking_error--------")
        # need to know a few shapes
        print("teach_times shape:", self.teach_times.shape)
        print("repeat_times shape:", self.repeat_times.shape)
        print("gps_teach_pose shape:", self.gps_teach_pose.shape)
        print("gps_repeat_pose shape:", self.gps_repeat_pose.shape)
        print("vtr_se2_pose shape:", self.vtr_se2_pose.shape)
        print("vtr_estimated_ptr shape:", self.vtr_estimated_ptr.shape)

        plt.figure()
        plt.title('PPK, VTR and Direct Estimated Path Tracking Error')
        # vtr estimate
        plt.plot(self.repeat_times,self.vtr_estimated_ptr, label=f'VTR RMSE: {np.sqrt(np.mean(self.vtr_estimated_ptr**2)):.3f}m for Repeat Max Error: {np.max(np.abs(self.vtr_estimated_ptr)):.3f}m')
        # ppk
        plt.plot(self.gps_repeat_pose[:,0], self.gps_ptr, label=f"PPK RMSE: {np.sqrt(np.mean(self.gps_ptr**2)):.3f}m for Repeat Max Error: {np.max(np.abs(self.gps_ptr)):.3f}m")
        # direct
        plt.plot(self.repeat_times, self.dir_ptr, label=f'Direct RMSE: {np.sqrt(np.mean(self.dir_ptr**2)):.3f}m for Repeat Max Error: {np.max(np.abs(self.dir_ptr)):.3f}m')
        plt.xlabel('Repeat Times')
        plt.ylabel('Path Tracking Error (m)')
        plt.grid()
        plt.legend()
        plt.show()

    def plot_localziation_error(self):
        print("--------In function plot_localization_error--------")

        print("vtr ptr shape:", self.vtr_estimated_ptr.shape)
        print("dir ptr shape:", self.dir_ptr.shape)
        print("gps ptr shape:", self.gps_ptr.shape)
        print("dro ptr shape:", self.dro_ptr.shape)

        self.vtr_estimated_ptr = -1* self.vtr_estimated_ptr.reshape(-1,1) # make it a column vector

        # also crop the dro path tracking 
        # self.dro_ptr[self.dro_ptr < -0.33] = 0 # this is a threshold to remove outliers   

        # yeah make it abs (optional)
        # self.vtr_estimated_ptr = np.abs(self.vtr_estimated_ptr)
        # self.dir_ptr = np.abs(self.dir_ptr)
        # self.gps_ptr = np.abs(self.gps_ptr)

        # I want to plot it in a 2 by 1
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title('VTR and Direct Estimated Localization Error')

        self.loc_error_vtr = []
        self.loc_error_dir = []
        self.loc_error_dro = []

        start_plot_idx = 1 # you can change when the plot starts

        for idx in range(0,self.vtr_estimated_ptr.shape[0]):
            repeat_vertex_time = self.repeat_times[idx]

            gps_idx = np.argmin(np.abs(self.gps_repeat_pose[:,0] - repeat_vertex_time))

            vtr_error = self.vtr_estimated_ptr[idx] - self.gps_ptr[gps_idx]
            self.loc_error_vtr.append(vtr_error)

        for idx in range(0,self.dir_ptr.shape[0]):
            repeat_vertex_time = self.repeat_times[idx]

            gps_idx = np.argmin(np.abs(self.gps_repeat_pose[:,0] - repeat_vertex_time))

            dir_error = self.dir_ptr[idx] - self.gps_ptr[gps_idx]
            self.loc_error_dir.append(dir_error)

        # for idx in range(0,self.dro_ptr.shape[0]):
        #     repeat_vertex_time = self.repeat_times[idx]

        #     gps_idx = np.argmin(np.abs(self.gps_repeat_pose[:,0] - repeat_vertex_time))

        #     dro_error = self.dro_ptr[idx] - self.gps_ptr[gps_idx]
        #     loc_error_dro.append(dro_error)
        
        self.loc_error_vtr = np.array(self.loc_error_vtr).reshape(-1,1)
        self.loc_error_dir = np.array(self.loc_error_dir).reshape(-1,1)
        # self.loc_error_dro = np.array(loc_error_dro).reshape(-1,1)



        # plt.plot(self.repeat_times[start_plot_idx:], loc_error_vtr[start_plot_idx:], label=f'VTR RMSE: {np.sqrt(np.mean(loc_error_vtr[start_plot_idx:]**2)):.3f}m for Repeat Max Error: {np.max(np.abs(loc_error_vtr[start_plot_idx:])):.3f}m')
        # # plt.plot(self.repeat_times[start_plot_idx:], loc_error_dir[start_plot_idx:], label=f'Direct RMSE: {np.sqrt(np.mean(loc_error_dir[start_plot_idx:]**2)):.3f}m for Repeat Max Error: {np.max(np.abs(loc_error_dir[start_plot_idx:])):.3f}m',color='green')
        # plt.plot(self.repeat_times[start_plot_idx:], loc_error_dro[start_plot_idx:], label=f'DRO RMSE: {np.sqrt(np.mean(loc_error_dro[start_plot_idx:]**2)):.3f}m for Repeat Max Error: {np.max(np.abs(loc_error_dro[start_plot_idx:])):.3f}m',color='orange')
        # plt.grid()
        # plt.legend()
        # plt.xlabel('Repeat Times')
        # plt.ylabel('Localization Error (m)')
        

        # plt.subplot(2, 1, 2)
        # # just plot the path tracking error

        # plt.title('VTR and DRO Estimated Path Tracking Error')
        # plt.plot(self.repeat_times[start_plot_idx:], self.vtr_estimated_ptr[start_plot_idx:], label=f'VTR RMSE: {np.sqrt(np.mean(self.vtr_estimated_ptr[start_plot_idx:]**2)):.3f}m for Repeat Max Error: {np.max(np.abs(self.vtr_estimated_ptr[start_plot_idx:])):.3f}m')
        # plt.plot(self.gps_repeat_pose[start_plot_idx:,0], self.gps_ptr[start_plot_idx:], label=f"PPK RMSE: {np.sqrt(np.mean(self.gps_ptr[start_plot_idx:]**2)):.3f}m for Repeat Max Error: {np.max(np.abs(self.gps_ptr[start_plot_idx:])):.3f}m")
        # plt.plot(self.repeat_times[start_plot_idx:], self.dir_ptr[start_plot_idx:], label=f'Direct RMSE: {np.sqrt(np.mean(self.dir_ptr[start_plot_idx:]**2)):.3f}m for Repeat Max Error: {np.max(np.abs(self.dir_ptr[start_plot_idx:])):.3f}m',color='green')
        # # plt.plot(self.repeat_times[start_plot_idx:], self.dro_ptr[start_plot_idx:], label=f'Direct RMSE: {np.sqrt(np.mean(self.dro_ptr[start_plot_idx:]**2)):.3f}m for Repeat Max Error: {np.max(np.abs(self.dro_ptr[start_plot_idx:])):.3f}m',color='orange')
        # plt.grid()
        # plt.legend()
        # plt.xlabel('Repeat Times')
        # plt.ylabel('Path Tracking Error (m)')
        # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # ## I will plot an identical one but with three errors
        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.title('Plot2: VTR and Direct Estimated Localization Error')
        plt.plot(self.repeat_times[start_plot_idx:], self.loc_error_vtr[start_plot_idx:], label=f'VTR RMSE: {np.sqrt(np.mean(self.loc_error_vtr[start_plot_idx:]**2)):.3f}m for Repeat Max Error: {np.max(np.abs(self.loc_error_vtr[start_plot_idx:])):.3f}m')
        plt.plot(self.repeat_times[start_plot_idx:], self.loc_error_dir[start_plot_idx:], label=f'Direct RMSE: {np.sqrt(np.mean(self.loc_error_dir[start_plot_idx:]**2)):.3f}m for Repeat Max Error: {np.max(np.abs(self.loc_error_dir[start_plot_idx:])):.3f}m',color='green')
        # plt.plot(self.repeat_times[start_plot_idx:], loc_error_dro[start_plot_idx:], label=f'DRO RMSE: {np.sqrt(np.mean(loc_error_dro[start_plot_idx:]**2)):.3f}m for Repeat Max Error: {np.max(np.abs(loc_error_dro[start_plot_idx:])):.3f}m',color='orange')
        plt.grid()
        plt.legend()
        plt.xlabel('Repeat Times')
        plt.ylabel('Localization Error (m)')
        plt.subplot(2, 1, 2)
        plt.title('Plot2: VTR and Direct Estimated Path Tracking Error')
        plt.plot(self.repeat_times[start_plot_idx:], self.vtr_estimated_ptr[start_plot_idx:], label=f'VTR RMSE: {np.sqrt(np.mean(self.vtr_estimated_ptr[start_plot_idx:]**2)):.3f}m for Repeat Max Error: {np.max(np.abs(self.vtr_estimated_ptr[start_plot_idx:])):.3f}m')
        plt.plot(self.gps_repeat_pose[start_plot_idx:,0], self.gps_ptr[start_plot_idx:], label=f"PPK RMSE: {np.sqrt(np.mean(self.gps_ptr[start_plot_idx:]**2)):.3f}m for Repeat Max Error: {np.max(np.abs(self.gps_ptr[start_plot_idx:])):.3f}m")
        plt.plot(self.repeat_times[start_plot_idx:], self.dir_ptr[start_plot_idx:], label=f'Direct RMSE: {np.sqrt(np.mean(self.dir_ptr[start_plot_idx:]**2)):.3f}m for Repeat Max Error: {np.max(np.abs(self.dir_ptr[start_plot_idx:])):.3f}m',color='green')
        # plt.plot(self.repeat_times[start_plot_idx:], self.dro_ptr[start_plot_idx:], label=f'DRO RMSE: {np.sqrt(np.mean(self.dro_ptr[start_plot_idx:]**2)):.3f}m for Repeat Max Error: {np.max(np.abs(self.dro_ptr[start_plot_idx:])):.3f}m',color='orange')
        plt.grid()
        plt.legend()
        plt.xlabel('Repeat Times')
        plt.ylabel('Path Tracking Error (m)')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gcf().set_size_inches(10, 10)
        plt.tight_layout()

        plt.show()


    def get_direct_path_tracking_error(self):
        print("--------In function get_direct_path_tracking_error--------")
        # need to get direct_path_tracking_error
        # step 1: make a path matrix
        # step 2: accumulate the signed distance
        dir_ptr = []
        previous_error = 0
        
        r_teach_world = []
        r_repeat_world = []
        r_repeat_world_vtr = []
        for idx in range(0,self.vtr_se2_pose.shape[0]):
            # the time stamps
            teach_vertex_time = self.teach_times[idx]
            repeat_vertex_time = self.repeat_times[idx]

            T_teach_world = self.teach_vertex_transforms[idx][0][teach_vertex_time[0]]
            T_gps_world_teach = T_novatel_robot @ T_teach_world
            r_gps_w_in_w_teach = T_gps_world_teach.r_ba_ina() # where the gps is in the world
            # r_teach_world_in_world = T_teach_world.r_ba_ina()
            r_teach_world.append(r_gps_w_in_w_teach.T)

            # direct result we can actually do everything in SE(3)
            r_repeat_teach_in_teach_se2 = self.direct_se2_pose[idx]
            # print("sam: this is direct estimate in se2: ",r_repeat_teach_in_teach_se2)

            # ok lets define the rotation and translation vector and then use the transformation matrix class
            def se2_to_se3(se2_vec):
                """
                Convert SE(2) pose to SE(3) transformation matrix
                
                Args:
                    se2_vec: Array-like of shape (3,) containing [x, y, theta] 
                            (theta in radians)
                
                Returns:
                    4x4 SE(3) transformation matrix as NumPy array
                """
                # Ensure input is flattened and extract components
                x, y, theta = np.asarray(se2_vec).flatten()
                
                # Create rotation matrix (Z-axis rotation)
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([
                    [c, -s,  0],
                    [s,  c,  0],
                    [0,  0,  1]
                ])
                
                # Create translation vector
                translation = np.array([x, y, 0])
                
                # Construct SE(3) matrix
                se3_matrix = np.eye(4)
                se3_matrix[:3, :3] = R          # Set rotation
                se3_matrix[:3, 3] = translation # Set translation
                
                return se3_matrix
            
            T_teach_repeat_direct = Transformation(T_ba = se2_to_se3(r_repeat_teach_in_teach_se2))
            # print("sam: this is direct estimate in se(3): \n",T_teach_repeat_direct.matrix())


            T_r_w = T_teach_repeat_direct.inverse() @ T_radar_robot @ T_teach_world # here there might a frame issue TODO

            T_gps_w_in_w_repeat = T_novatel_robot @ T_radar_robot.inverse() @ T_r_w # here there might be a frame issue TODO I think this is correct
            # r_r_w_in_world = T_r_w.r_ba_ina().T

            r_gps_w_in_w_repeat = T_gps_w_in_w_repeat.r_ba_ina() # where the gps is in the world
            # print("sam: direct r_gps_w_in_w_repeat: \n", r_gps_w_in_w_repeat)

            # print("r_r_w_in_world shape:", r_r_w_in_world.shape)
            # print("r_r_w_in_world:", r_r_w_in_world.T[0:2])

            # print("double check:", r_gps_w_in_w_repeat[0:2].shape)
            r_repeat_world.append(r_gps_w_in_w_repeat[0:2].T)

            # need to investigate vtr_repeat (TODO) seems to be on the other side of the teach compared to direct 
            T_teach_repeat_edge = self.repeat_edge_transforms[idx][0][repeat_vertex_time[0]]

            # print("sam: this is vtr estimate se(3): \n",T_teach_repeat_edge.matrix())

            T_repeat_w = T_teach_repeat_edge.inverse() @ T_teach_world
            T_gps_w_in_w_repeat_vtr = T_novatel_robot @ T_repeat_w
            r_gps_w_in_w_repeat_vtr = T_gps_w_in_w_repeat_vtr.r_ba_ina()
           

            r_repeat_world_vtr.append(r_gps_w_in_w_repeat_vtr.T)
            # print("sam: vtr r_gps_w_in_w_repeat: \n", r_gps_w_in_w_repeat_vtr)

        # make them into numpy arrays
        self.teach_world = np.array(r_teach_world).reshape(-1,3)
        self.repeat_world_direct = np.array(r_repeat_world).reshape(-1,2)
        self.repeat_world_vtr = np.array(r_repeat_world_vtr).reshape(-1,3)

        # plt.figure()
        # plt.title('Teach and Direct Repeat World')
        # plt.plot(self.teach_world[:,0], self.teach_world[:,1], label='RTR Teach World in GPS', linewidth = 1)
        # plt.scatter(self.repeat_world_direct[:,0], self.repeat_world_direct[:,1], label='Direct Repeat World in GPS', s=10,marker='o',color="green")
        # plt.scatter(self.repeat_world_vtr[:,0], self.repeat_world_vtr[:,1], label='VTR Repeat World in GPS', s=10,marker='x')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')
        # plt.grid()
        # plt.legend()
        # plt.axis('equal')
        # plt.show()

        # now we are ready to set up the path matrix
        self.teach_world[:,2] = 0 # we are in a plane otherwise need to change line 232 to the height of the gps
        path_matrix = path_to_matrix(self.teach_world) 
        print("in function get_direct_path_tracking_error the shape of self.teach_world is:", self.teach_world.shape)

        # self.repeat world shape (:,2)

        for idx in range(0,self.repeat_world_direct.shape[0]):
            # print("-----idx:", idx)
            repeat_x_y_z = np.array([self.repeat_world_direct[idx][0], self.repeat_world_direct[idx][1], 0]).T # this is how tall the gps is

            # print("repeat_x_y_z shape:", repeat_x_y_z.shape)
            # get the signed distance to the path
            error = signed_distance_to_path(repeat_x_y_z,path_matrix)

            product = error*previous_error
            if product<0 and (abs(error)>0.10 and abs(previous_error)>0.10):
                error = -1* error
            # print("error:", error)
            dir_ptr.append(error)
            previous_error = error
        
        dir_ptr = np.array(dir_ptr).reshape(-1,1) # n by 1
        return dir_ptr
    

    def get_dro_path_tracking_error(self):
        print("--------In function get_dro_path_tracking_error--------")
        # need to get dro_path_tracking_error
        # step 1: make a path matrix
        # step 2: accumulate the signed distance
        z_dummy = np.zeros_like(self.r_teach_world_dro[:,0]) # we need to make sure the z is 0
        self.r_teach_world_dro = np.hstack((self.r_teach_world_dro, z_dummy.reshape(-1,1)))
        self.r_repeat_world_dro = np.hstack((self.r_repeat_world_dro, z_dummy.reshape(-1,1)))

        print("r_teach_world_dro shape:", self.r_teach_world_dro.shape)
        path_matrix = path_to_matrix(self.r_teach_world_dro)

        dro_ptr = []
        previous_error = 0
        for idx in range(0,self.r_repeat_world_dro.shape[0]):
            # get the signed distance to the path
            error = signed_distance_to_path(self.r_repeat_world_dro[idx],path_matrix)

            product = error*previous_error
            if product<0 and (abs(error)>0.10 and abs(previous_error)>0.10):
                error = -1* error
            dro_ptr.append(error)
            previous_error = error
        return np.array(dro_ptr).reshape(-1,1) # n by 1


    def get_gps_path_tracking_error(self):
        print("--------In function get_gps_path_tracking_error--------")
        # need to get gps_path_tracking_error
        # step 1: make a path matrix
        # step 2: accumulate the signed distance
        
        gps_teach_pose_without_time = self.gps_teach_pose[:,1:4] # m by 3
        gps_repeat_pose_without_time  = self.gps_repeat_pose[:,1:4] # n by 3

        print("gps_teach_pose_without_time shape:", gps_teach_pose_without_time.shape)
        print("gps_repeat_pose_without_time shape:", gps_repeat_pose_without_time.shape)

        path_matrix = path_to_matrix(gps_teach_pose_without_time)
        
        gps_ptr = []
        previous_error = 0
        for idx in range(0,self.gps_repeat_pose.shape[0]):
            # get the signed distance to the path
            error = signed_distance_to_path(gps_repeat_pose_without_time[idx],path_matrix)

            product = error*previous_error
            if product<0 and (abs(error)>0.07 and abs(previous_error)>0.07): # this value can be tuned
                error = -1 * error
            gps_ptr.append(error)
            previous_error = error

        return np.array(gps_ptr).reshape(-1,1) # n by 1



    def plot_traj(self, teach_array, repeat_array):
        print("--------In function plot_traj--------")
        # need to plot the gps and ppk
        plt.figure()
        # plt.title('Teach and Repeat World')
        plt.plot(teach_array[:,0], teach_array[:,1], label='RTR Teach World in GPS', linewidth = 1)
        plt.scatter(repeat_array[:,0], repeat_array[:,1], label='RTR Repeat World in GPS', s=3,marker='o',color="green")
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid()
        plt.legend()
        plt.axis('equal')
        # plt.show()

        return True
    
    def show_plots(self):
        plt.show()
        return True




if __name__ == "__main__":
    path_to_data = "/home/samqiao/ASRL/vtr3_testing/scripts/direct/grassy_t2_r3"

    plotter = Plotter()

    plotter.set_data(path_to_data)

    # plotter.plot()

    plotter.plot_localziation_error()

    # plotter.show_plots()

