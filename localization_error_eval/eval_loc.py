import os
import os.path as osp
import argparse
import numpy as np
import numpy.linalg as npla
import csv

import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

from pyboreas import BoreasDataset
from pylgmath import se3op

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Posegraph dependency
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory

from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path

# Sam helper functions
from helper import *


# FLAGs
DEBUG = True
PLOT = True


# static transform
T_radar_robot = np.array([[1, 0, 0, -0.025],
                          [0, -1, 0,-0.002],
                          [0, 0, -1, 1.03218],
                          [0, 0, 0, 1]])

T_gps_lidar = np.array([[0, -1, 0, 0.077],
                          [0, 0, -1, -0.178],
                          [1, 0, 0, 0.575],
                          [0, 0, 0, 1]])

T_radar_lidar = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0.153],
                          [0, 0, 0, 1]])

T_radar_gps = T_radar_lidar @ get_inverse_tf(T_gps_lidar) 

T_robot_gps = get_inverse_tf(T_radar_robot) @ T_radar_gps

if DEBUG:
    print("T_radar_robot: ", T_radar_robot)
    print("T_gps_lidar: ", T_gps_lidar)
    print("T_radar_lidar: ", T_radar_lidar)
    print("T_radar_gps: ", T_radar_gps)
    print("T_robot_gps: ", T_robot_gps)

class BagFileParser():
  def __init__(self, bag_file):
    try:
      self.conn = sqlite3.connect(bag_file)
    except Exception as e:
      print('Could not connect: ', e)
      raise Exception('could not connect')

    self.cursor = self.conn.cursor()

    ## create a message (id, topic, type) map
    topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()

    self.topic_type = {name_of: type_of for id_of, name_of, type_of in topics_data}
    self.topic_id = {name_of: id_of for id_of, name_of, type_of in topics_data}
    self.topic_msg_message = {name_of: get_message(type_of) for id_of, name_of, type_of in topics_data}

  # Return messages as list of tuples [(timestamp0, message0), (timestamp1, message1), ...]
  def get_bag_messages(self, topic_name):
    topic_id = self.topic_id[topic_name]
    rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
    return [(timestamp, deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp, data in rows]
  

def main(dataset_dir, result_dir):
  result_dir = osp.normpath(result_dir)
  odo_input = osp.basename(result_dir)
  loc_inputs = [i for i in os.listdir(result_dir) if (i != odo_input and i.startswith("2024"))]
  loc_inputs.sort()
  print("Result Directory:", result_dir)
  print("Odometry Run:", odo_input)
  print("Localization Runs:", loc_inputs)
  print("Dataset Directory:", dataset_dir)

  repeat_posegraph_path = os.path.join(result_dir, loc_inputs[0])

  # lets do one repeat for now in the future we can do multiple repeats

  factory = Rosbag2GraphFactory(os.path.join(repeat_posegraph_path,"graph"))

  repeat_graph = factory.buildGraph()

  nruns =  g_utils.count_runs(repeat_graph)

  if DEBUG:
    print("Graph info: ")
    print("repeat_posegraph_path: ", repeat_posegraph_path)
    print(f"Graph {repeat_graph} has {repeat_graph.number_of_vertices} vertices and {repeat_graph.number_of_edges} edges")
    print(f"Graph {repeat_graph} has {nruns} runs")

  g_utils.set_world_frame(repeat_graph, repeat_graph.root)
  v_start_teach = repeat_graph.root

  print("pose v_start_teach: ",v_start_teach.T_v_w.r_ba_ina())
  print("v_start_teach_stamp: ",v_start_teach.stamp)

  path_matrix = vtr_path.path_to_matrix(repeat_graph, PriviledgedIterator(v_start_teach))

  x_teach = []
  y_teach = []
  t_teach = []

  for v, e in PriviledgedIterator(v_start_teach):
        estimated_teach_pose_in_robot_frame = v.T_v_w.r_ba_ina()

        # print("first:", estimated_teach_pose_in_robot_frame)

        estimated_teach_pose_in_gps_frame = T_robot_gps @ np.vstack((estimated_teach_pose_in_robot_frame,[1]))

        # print(T_robot_gps[0:3,0:3])

        # print("second: ",estimated_teach_pose_in_gps_frame)

        x_teach.append(estimated_teach_pose_in_gps_frame[1])
        y_teach.append(estimated_teach_pose_in_gps_frame[2])
        t_teach.append(v.stamp / 1e9)


# Those coordinates are in the GPS frame
  x_teach = np.array(x_teach)
  y_teach = np.array(y_teach)
  t_teach = np.array(t_teach).reshape(-1,1)
  t_teach_offset = t_teach[0]
  # reset to zero
  t_teach = np.squeeze(t_teach - t_teach_offset)

  # print(x_teach.shape)
  # print(y_teach.shape)
  # print(t_teach.shape)
  # path_teach = np.hstack((x_teach, y_teach,t_teach)).T

  # print("Sam:", path_teach.shape)

  # now for the repeat branch
  v_start_repeat = repeat_graph.get_vertex((1,0))

  print("pose v_start_repeat: ",v_start_repeat.T_v_w.r_ba_ina())
  print("v_start_repeat_stamp: ",v_start_repeat.stamp)

  x_repeat = []
  y_repeat = []
  t_repeat = []
  proj_dist = []
  path_len = 0

  for v, e in TemporalIterator(v_start_repeat):
        
        closest_teach_vertex = g_utils.get_closest_teach_vertex(v)
        # print("repeat vertex timestamp:",v.stamp)
        # print("closest teach timestamp:",closest_teach_vertex.stamp)
        
        estimated_repeat_pose_in_robot_frame = v.T_v_w.r_ba_ina()
        estimated_repeat_pose_in_gps_frame = T_robot_gps @ np.vstack((estimated_repeat_pose_in_robot_frame,[1]))

        x_repeat.append(estimated_repeat_pose_in_gps_frame[1])
        y_repeat.append(estimated_repeat_pose_in_gps_frame[2])
        t_repeat.append(v.stamp / 1e9)
        proj_dist.append([vtr_path.distance_to_path(v.T_v_w.r_ba_ina(), path_matrix),v.stamp / 1e9,closest_teach_vertex.stamp / 1e9])
        path_len += np.linalg.norm(e.T.r_ba_ina())

  x_repeat = np.array(x_repeat)
  y_repeat = np.array(y_repeat)
  t_repeat = np.array(t_repeat).reshape(-1,1)
  t_repeat_offset = t_repeat[0]
  # reset to zero
  t_repeat = np.squeeze(t_repeat - t_repeat_offset)

  proj_dist = np.array(proj_dist)
  
  # path_repeat = np.hstack((x_repeat, y_repeat,t_repeat)).T

  # estimated_error = get_estimated_error(path_teach, path_repeat) 

  # print("estimated error shape:", estimated_error.shape)
  # print("proj_dist shape:", proj_dist.shape)

  # now we deal with the gps data
  teach_rosbag_path = os.path.join(dataset_dir, odo_input)
  repeat_rosbag_path = os.path.join(dataset_dir, loc_inputs[0])

  x_teach_gps, y_teach_gps, t_teach_gps = get_xy_gps(teach_rosbag_path)
  x_gps_offset = x_teach_gps[0]
  y_gps_offset = y_teach_gps[0]
  t_gps_teach_0 = t_teach_gps[0]
  # reset to zero
  t_teach_gps = t_teach_gps - t_gps_teach_0

  # print(t_teach[0:10])
  # print(t_teach_gps[0:10])

  x_repeat_gps, y_repeat_gps, t_repeat_gps = get_xy_gps(repeat_rosbag_path)
  # # subtract the origin
  # # because I am repeating backwards temporary TODO
  # x_repeat_gps = x_repeat_gps[::-1]
  # y_repeat_gps = y_repeat_gps[::-1]
  # t_repeat_gps = t_repeat_gps[::-1]
  # t_repeat_last = t_repeat_gps[-1]
  # reset to zero
  t_repeat_gps_0 = t_repeat_gps[0]
  t_repeat_gps = t_repeat_gps - t_repeat_gps_0

  if DEBUG:
    print("GPS debug:: \n")
    print("t_teach: ",t_teach[-1])
    print("t_teach_gps: ",t_teach_gps[-1])
    print("t_repeat: ",t_repeat[-1])
    print("t_repeat_gps: ",t_repeat_gps[-1])
    print("------- line break ------ \n")
    print("t_teach shape: ",t_teach.shape)
    print("t_repeat shape: ",t_repeat.shape)
    print("t_teach_gps shape: ",t_teach_gps.shape)
    print("t_repeat_gps shape: ",t_repeat_gps.shape)

  # we need to create a dictionary of the teach and repeat gps data
  # where the key would be the timestamp and the value would be the x,y coordinates
  # we can then use this dictionary to align the timestamps of the teach and repeat gps data
  # lets start with the teach gps data
  teach_gps_dict = dict()
  for i in range(0,len(t_teach_gps)):
    teach_gps_dict[t_teach_gps[i]] = (x_teach_gps[i],y_teach_gps[i])

  repeat_gps_dict = dict()
  for i in range(0,len(t_repeat_gps)):
    repeat_gps_dict[t_repeat_gps[i]] = (x_repeat_gps[i],y_repeat_gps[i])

  # we can do it with the vertex timestamps
  teach_vtr_dict = dict()
  for i in range(0,len(t_teach)):
    teach_vtr_dict[t_teach[i]] = (x_teach[i],y_teach[i])
  
  repeat_vtr_dict = dict()
  for i in range(0,len(t_repeat)):
    repeat_vtr_dict[t_repeat[i]] = (x_repeat[i],y_repeat[i])
  

  localization_error = []
  # lets go through all the repeat vertices 
  vertex_cnt = 0
  v_start_repeat = repeat_graph.get_vertex((1,0))
  for v, e in TemporalIterator(v_start_repeat):      
        print("Processing new repeat vertex: ------------------- ",vertex_cnt)
        repeat_vertex = v
        repeat_vertex_timestamp = (repeat_vertex.stamp/1e9) - t_repeat_offset

        print("repeat_vertex_timestamp: ",repeat_vertex_timestamp[0])

        repeat_x, repeat_y = repeat_vtr_dict[repeat_vertex_timestamp[0]]

        # print("Sam debug repeat: ",repeat_vertex_timestamp,repeat_x,repeat_y)

        # get the closest teach vertex
        closest_teach_vertex = g_utils.get_closest_teach_vertex(v)
        closest_teach_vertex_timestamp = (closest_teach_vertex.stamp / 1e9)-t_teach_offset
        print("closest_teach_vertex_timestamp: ",closest_teach_vertex_timestamp[0])
        teach_x, teach_y = teach_vtr_dict[closest_teach_vertex_timestamp[0]]
        # print("Sam debug teach: ",closest_teach_vertex_timestamp,teach_x,teach_y)

        estimated_vtr_distance = np.linalg.norm(np.array([repeat_x,repeat_y]) - np.array([teach_x,teach_y]))
        print("estimated_vtr_distance: ",estimated_vtr_distance)

        # lets get the gps data
        repeat_gps_timestamp = get_closest_timestamp(repeat_vertex_timestamp[0],t_repeat_gps)
        print("repeat_gps_timestamp: ",repeat_gps_timestamp)
        repeat_gps_x, repeat_gps_y = repeat_gps_dict[repeat_gps_timestamp]
        print("repeat_gps_x, repeat_gps_y: ",repeat_gps_x,repeat_gps_y)

        teach_gps_timestamp = get_closest_timestamp(closest_teach_vertex_timestamp[0],t_teach_gps)
        print("teach_gps_timestamp: ",teach_gps_timestamp)
        teach_gps_x, teach_gps_y = teach_gps_dict[teach_gps_timestamp]
        print("teach_gps_x, teach_gps_y: ",teach_gps_x,teach_gps_y)

        estimated_gps_distance = np.linalg.norm(np.array([repeat_gps_x,repeat_gps_y]) - np.array([teach_gps_x,teach_gps_y]))
        print("estimated_gps_distance: ",estimated_gps_distance)

        error = estimated_gps_distance - estimated_vtr_distance
        if abs(error) >3:
           print("error exceed 3m at vertex cnt",vertex_cnt)
        print("loc error: ",error)
        localization_error.append(error)
        vertex_cnt += 1
        print("Finshed! ------------------- \n ")


  
  localization_error = np.array(localization_error)
  print("localization_error shape: ",localization_error.shape)
  print("localization_error_mean: ",np.mean(localization_error))

  if PLOT:
    fs = 30
    plt.figure(0)
    plt.scatter(x_teach, y_teach, label="Teach")
    plt.scatter(x_repeat, y_repeat, label="Repeat")
    # plt.scatter(x_teach_gps, y_teach_gps, label="Teach GPS")
    # plt.scatter(x_repeat_gps, y_repeat_gps, label="Repeat GPS")
    plt.axis('equal')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Teach and Repeat Path from PoseGraph")
    plt.legend(fontsize=fs)


    fig = plt.figure(figsize =(10, 7))
    # Creating plot
    plt.boxplot(localization_error,autorange=True)
    plt.axis('equal')
    plt.title("Localization Error Boxplot",fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    # show plot
    plt.show()
  
  print(f"Path {1} was {path_len:.3f}m long")

  return True
  








if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # Assuming following path structure:
  # <rosbag name>/metadata.yaml
  # <rosbag name>/<rosbag name>_0.db3
  parser.add_argument('--dataset', default=os.getcwd(), type=str, help='path to boreas dataset (contains boreas-*)')
  parser.add_argument('--path', default=os.getcwd(), type=str, help='path to vtr folder (default: os.getcwd())')

  args = parser.parse_args()

  main(args.dataset, args.path)
