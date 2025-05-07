import os
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

# from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
# from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator
# import vtr_pose_graph.graph_utils as g_utils
# import vtr_regression_testing.path_comparison as vtr_path
# import argparse

import sys
parent_folder = "/home/samqiao/ASRL/vtr3_testing"

# Insert path at index 0 so it's searched first
sys.path.insert(0, parent_folder)

from deps.path_tracking_error.fcns import *

# from radar.utils.helper import *

# # point cloud vis
# from sensor_msgs_py.point_cloud2 import read_points
# # import open3d as o3d
from pylgmath import Transformation
# from vtr_utils.plot_utils import *
# import time

import yaml
import gp_doppler as gpd
import torch
import torchvision

# from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images

# from process_vtr import get_vtr_ptr_baseline

from utils import *
from plotter import Plotter


# print("Current working dir", os.getcwd())

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

config = load_config(os.path.join(parent_folder,'scripts/direct/direct_config.yaml'))


db_bool = config['bool']
SAVE = db_bool.get('SAVE')
SAVE = False
print("SAVE:",SAVE)
PLOT = db_bool.get('PLOT')
DEBUG = db_bool.get('DEBUG')

result_folder = config.get('output')

# change here
out_path_folder = os.path.join(result_folder,f"grassy_t2_r3/")
if not os.path.exists(out_path_folder):
    os.makedirs(out_path_folder)
    print(f"Folder '{out_path_folder}' created.")
else:
    print(f"Folder '{out_path_folder}' already exists.")    


sequence = "grassy_t2_r3"

sequence_path = os.path.join(result_folder, sequence)
if not os.path.exists(sequence_path):
    print("ERROR: No sequence found in " + sequence_path)
    exit(0)

TEACH_FOLDER = os.path.join(sequence_path, "teach")
REPEAT_FOLDER = os.path.join(sequence_path, "repeat")
RESULT_FOLDER = os.path.join(sequence_path, "direct")

if not os.path.exists(TEACH_FOLDER):
    raise FileNotFoundError(f"Teach folder {TEACH_FOLDER} does not exist.")
if not os.path.exists(REPEAT_FOLDER):
    raise FileNotFoundError(f"Repeat folder {REPEAT_FOLDER} does not exist.")
if not os.path.exists(RESULT_FOLDER):
    raise FileNotFoundError(f"Result folder {RESULT_FOLDER} does not exist.")

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


# load the repeat data
repeat_df = np.load(os.path.join(REPEAT_FOLDER, f"repeat.npz"),allow_pickle=True)
# in the repeat
repeat_times = repeat_df['repeat_times']
repeat_polar_imgs = repeat_df['repeat_polar_imgs']
repeat_azimuth_angles = repeat_df['repeat_azimuth_angles']
repeat_azimuth_timestamps = repeat_df['repeat_azimuth_timestamps']
repeat_vertex_timestamps = repeat_df['repeat_vertex_timestamps']
repeat_edge_transforms = repeat_df['repeat_edge_transforms']
vtr_estimated_ptr = repeat_df['dist']


# load the result data
result_df = np.load(os.path.join(RESULT_FOLDER, f"result.npz"),allow_pickle=True)
vtr_norm = result_df['vtr_norm']
gps_norm = result_df['gps_norm']
dir_norm = result_df['dir_norm']
direct_se2_pose = result_df['direct_se2_pose']
vtr_se2_pose = result_df['vtr_se2_pose']
gps_teach_pose = result_df['gps_teach_pose']
gps_repeat_pose = result_df['gps_repeat_pose']

print("gps_teach_pose", gps_teach_pose.shape)

# plotter = Plotter()
# plotter.plot_traj(gps_teach_pose[:,1:],gps_repeat_pose[:,1:])
# plotter.show_plots()


# actually mtplotlib inherently supports animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Animator:
    def __init__(self):
        # self.fig = []
        # self.fig.append(plt.figure())
        # self.ax = plt.axes(projection='2d'
        self.fig, self.ax = plt.subplots()  # Cleaner 2D setup

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

        # # gps ptr
        # self.gps_ptr = self.get_gps_path_tracking_error()
        # print("gps_ptr shape:", self.gps_ptr.shape)

        # # direct ptr
        # self.dir_ptr = self.get_direct_path_tracking_error()
        # print("dir_ptr shape:", self.dir_ptr.shape)


        # get the teachworld, repeat world, and the vtr world
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

        print("teach world shape:", self.teach_world.shape)
        print("repeat world shape:", self.repeat_world_direct.shape)
        print("repeat world vtr shape:", self.repeat_world_vtr.shape)


        # deepseek
        # Plot static elements once (the entire teach path)
        self.ax.plot(self.teach_world[:, 0], self.teach_world[:, 1], 
                    label='Teach World in GPS', linewidth=1, color='blue')
        
        # Initialize dynamic elements (scatter plots)
        self.direct_scatter = self.ax.scatter([], [], s=5, marker='X', 
                                            color='green', label='Direct Repeat', zorder = 3,alpha=0.8)
        self.vtr_scatter = self.ax.scatter([], [], s=5, marker='X', 
                                         color='red', label='VTR Repeat',zorder = 2)
        
        self.current_direct = self.ax.scatter(
            [], [], 
            s=50,           # Larger size (default was 30)
            linewidths=1.0, # Thinner marker edges
            color='green',
            alpha=0.8,
            marker='x',
            zorder=3,       # Ensure markers stay on top
            edgecolors='black'  # Optional: Add edge contrast
        )
        
        self.current_vtr = self.ax.scatter(
            [], [],
            s=50, 
            linewidths=1.0,
            color='red',
            marker='x',     # X marker
            alpha=1.0,
            zorder=2
        )
        
        # Configure axis properties
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title('Teach and Direct Repeat World')
        self.ax.grid(True)
        self.ax.legend()
        
        # Set equal aspect ratio with box adjustment
        self.ax.set_aspect('equal', adjustable='box')  # Critical fix

    def add_fig(self,figure):
        self.fig.append(figure)
        return True

    def get_fig(self,index):
        return self.fig[index]

    def update(self, i):
        print("------ processing frame:", i, "------")
        self.direct_scatter.set_offsets(self.repeat_world_direct[:i+1, :2])
        self.vtr_scatter.set_offsets(self.repeat_world_vtr[:i+1, :2])
        
        # Update current pose markers (single point at current position)
        self.current_direct.set_offsets([self.repeat_world_direct[i, :2]])
        self.current_vtr.set_offsets([self.repeat_world_vtr[i, :2]])
        
        # Update view window
        current_x = self.teach_world[i, 0]
        current_y = self.teach_world[i, 1]
        self.ax.set_xlim(current_x - 15, current_x + 15)
        self.ax.set_ylim(current_y - 15, current_y + 15)
        
        return (self.direct_scatter, self.vtr_scatter, 
                self.current_direct, self.current_vtr)




    def plot(self):
        plt.figure()
        plt.title('Teach and Direct Repeat World')
        plt.plot(self.teach_world[:,0], self.teach_world[:,1], label='RTR Teach World in GPS', linewidth = 1)
        plt.scatter(self.repeat_world_direct[:,0], self.repeat_world_direct[:,1], label='Direct Repeat World in GPS', s=10,marker='o',color="green")
        plt.scatter(self.repeat_world_vtr[:,0], self.repeat_world_vtr[:,1], label='VTR Repeat World in GPS', s=10,marker='x')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid()
        plt.legend()
        plt.axis('equal')
        plt.show()


    def animation(self, fps=120):
        frames = int(self.repeat_times.shape[0])
        print("frames:", frames)
        self.ani = FuncAnimation(self.fig, self.update, frames = frames, blit=True,
        interval=1000/fps)
        # plt.show()


    def save(self, path):
        from matplotlib.animation import FFMpegWriter
        # Configure AVI writer
        writer = FFMpegWriter(
            fps=40,
            codec='mpeg4',  # Or 'libx264' for H.264
            bitrate=5000,
            extra_args=['-crf', '18']
        )
        
        # Create animation
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            frames=int(self.repeat_times.shape[0]),
            blit=True
        )
        
        # Save AVI
        self.ani.save(
            os.path.join(path, 'parking_estimates.mp4'),
            writer=writer,
            dpi=100
        )

if __name__ == "__main__":

    out_path_folder = os.path.join(result_folder,f"grassy_t2_r3/")
    # load the data
    animator = Animator()
    animator.set_data(sequence_path)

    animator.animation(fps=120)

    # animator.animation()
    animator.save(out_path_folder)
