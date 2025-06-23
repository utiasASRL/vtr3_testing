import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs_py.point_cloud2 import read_points
import open3d as o3d
import sys
from vtr_utils.plot_utils import extract_map_from_vertex
import argparse
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator, SpatialIterator
import vtr_pose_graph.graph_utils as g_utils

sys.path.append('/home/samqiao/ASRL/vtr3_testing/src/vtr_pose_graph/src')

if __name__ == '__main__':

  parser = argparse.ArgumentParser(prog = 'Plot Point Clouds Path',
              description = 'Plots point clouds')
  parser.add_argument('-g', '--graph', default=os.getenv("VTRDATA"), help="The filepath to the pose graph folder. (Usually /a/path/graph)")
  parser.add_argument('--save_pc', type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to save point clouds (True/False).")
  parser.add_argument('--save_dir', type=str, default=None, help="Directory to save point clouds (required if --save_pc is True).")
  args = parser.parse_args()

  if args.save_pc and not args.save_dir:
    parser.error("--save_dir must be specified if --save_pc is True.")

  if args.save_pc and not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

  factory = Rosbag2GraphFactory(args.graph)

  test_graph = factory.buildGraph()
  print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

  g_utils.set_world_frame(test_graph, test_graph.root)
  v_start = test_graph.root

  x = []
  y = []
  live_2_map = []
  map_2_live = []

  first = True
  paused = False
  def toggle(vis):
    global paused
    paused = not paused
    return False

  vis = o3d.visualization.VisualizerWithKeyCallback()
  vis.register_key_callback(ord(' '), toggle)
  vis.create_window()

  pcd = o3d.geometry.PointCloud()
  vis.poll_events()
  vis.update_renderer()

  cloud_count = 0

  for i in range(test_graph.major_id + 1):
    v_start = test_graph.get_vertex((i, 0))
    paused = True
    vertices = list(TemporalIterator(v_start))
    vertices_to_plot = vertices[:-10] if len(vertices) > 10 else vertices

    # Accumulate all points for this major_id
    accumulated_points = []

    for idx, (vertex, e) in enumerate(vertices_to_plot):

      new_points, map_ptr = extract_map_from_vertex(test_graph, vertex)

      num_points = new_points.shape[1]
      print(f'Number of points: {num_points}')

      robot_position = vertex.T_v_w.r_ba_ina().reshape((3,) )
      print('robot position = ', robot_position)
      robot_pose = vertex.T_v_w.matrix()
      print('robot_pose = ', vertex.T_v_w.matrix())

      x.append(vertex.T_v_w.r_ba_ina()[0]) 
      y.append(vertex.T_v_w.r_ba_ina()[1])

      pcd.points = o3d.utility.Vector3dVector(new_points.T)
      if np.allclose(map_ptr.matrix(), np.eye(4)):
        pcd.paint_uniform_color((1.0, 0.0, 0.0)) # Red color for identity matrix
      else:
        pcd.paint_uniform_color((0.1*vertex.run, 0.25*vertex.run, 0.45))

      # Save every 20th point cloud if requested
      if args.save_pc and (cloud_count % 20 == 0):
        filename = os.path.join(args.save_dir, f"cloud_{cloud_count:05d}.ply")
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved point cloud to {filename}")

      # Accumulate all points for this major_id
      if args.save_pc:
        accumulated_points.append(new_points.T)

      cloud_count += 1

      # Create coordinate frame for the vertex
      frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=robot_position)

      if first:
        first = False
        vis.add_geometry(pcd)
        vis.add_geometry(frame)
      else:
        vis.update_geometry(pcd)
        vis.remove_geometry(frame, reset_bounding_box=False)
        vis.add_geometry(frame)
      t = time.time()
      while time.time() - t < 0.1 or paused:
        vis.poll_events()
        vis.update_renderer()

    # Save accumulated point cloud for this major_id
    if args.save_pc and accumulated_points:
      all_points = np.vstack(accumulated_points)
      acc_pcd = o3d.geometry.PointCloud()
      acc_pcd.points = o3d.utility.Vector3dVector(all_points)
      acc_filename = os.path.join(args.save_dir, f"accumulated_major_{i:03d}.ply")
      o3d.io.write_point_cloud(acc_filename, acc_pcd)
      print(f"Saved accumulated point cloud for major_id {i} to {acc_filename}")

  print("Finished processing all point clouds.")

  vis.run()
  vis.destroy_window()