import os
import matplotlib.pyplot as plt
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory

from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Plot Repeat Path',
                        description = 'Plots scatter of points to show path. Also calculates RMS error')
    parser.add_argument('-g', '--graph', default=os.getenv("VTRDATA"), help="The filepath to the pose graph folder. (Usually /a/path/graph)")
    parser.add_argument('-f', '--filter', type=int, nargs="*", help="Select only some of the repeat runs. Default plots all runs.")
    args = parser.parse_args()

    offline_graph_dir = args.graph
    factory = Rosbag2GraphFactory(offline_graph_dir)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

    g_utils.set_world_frame(test_graph, test_graph.root)

    v_start = test_graph.root

    path_matrix = vtr_path.path_to_matrix(test_graph, PriviledgedIterator(v_start))
    print(path_matrix.shape)

    x = []
    y = []
    t = []

    for v, e in PriviledgedIterator(v_start):
        x.append(v.T_v_w.r_ba_ina()[0])
        y.append(v.T_v_w.r_ba_ina()[1])
        t.append(v.stamp / 1e9)

    plt.figure(0)
    plt.scatter(x, y, label="Teach")
    plt.axis('equal')

    if args.filter is None:
        args.filter = [i+1 for i in range(test_graph.major_id)]

    for i in range(test_graph.major_id):
        if i+1 not in args.filter:
            continue
        v_start = test_graph.get_vertex((i+1,0))

        x = []
        y = []
        t = []
        dist = []
        path_len = 0

        for v, e in TemporalIterator(v_start):
            x.append(v.T_v_w.r_ba_ina()[0])
            y.append(v.T_v_w.r_ba_ina()[1])
            t.append(v.stamp / 1e9)
            dist.append(vtr_path.signed_distance_to_path(v.T_v_w.r_ba_ina(), path_matrix))
            path_len += np.linalg.norm(e.T.r_ba_ina())
        
        print(f"Path {i+1} was {path_len:.3f}m long with {len(x)} vertices")
        if len(t) < 2:
            continue

        plt.figure(0)
        plt.scatter(x, y, label=f"Repeat {i+1}")
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.legend()

        plt.figure(1)
        plt.title(f"Position for Repeat {i+1}")
        plt.plot(t, x, label="X")
        plt.plot(t, y, label="Y")
        plt.legend()
        plt.ylabel("Position (m)")
        plt.xlabel("Time (s)")

        plt.figure(2)
        rmse_int = np.sqrt(np.trapz(np.array(dist)**2, t) / (t[-1] - t[0]))
        rmse = np.sqrt(np.mean(np.array(dist)**2))
        print(rmse, rmse_int)

        t_duplicate = np.array(t.copy())
        t_duplicate = t_duplicate - t_duplicate[0]
        plt.plot(t_duplicate, dist, label=f"RMSE: {rmse:.3f}m for Repeat {i+1}")
        plt.legend()
        plt.ylabel("Path Tracking Error (m)")
        plt.xlabel("Time (s)")
        plt.title("Path Tracking Error")
        plt.grid()
   
        plt.figure(3)
        x = np.array(x).squeeze()
        y = np.array(y).squeeze()
        vx = np.gradient(x, t).squeeze()
        vy = np.gradient(y, t).squeeze()
        plt.plot(t, vx, label=f"Vx for Repeat {i+1}")
        plt.plot(t, vy, label=f"Vy for Repeat {i+1}")
        plt.plot(t, np.hypot(vx, vy), label=f"V for Repeat {i+1}")
        plt.legend()
        plt.ylabel("Velocity (m/s)")
        plt.xlabel("Time (s)")

        plt.figure(4)
        acc_x = np.gradient(vx, t).squeeze()
        acc_y = np.gradient(vy, t).squeeze()
        plt.plot(t, acc_x, label=f"Acc_x for Repeat {i+1}")
        plt.plot(t, acc_y, label=f"Acc_y for Repeat {i+1}")
        plt.plot(t, np.hypot(acc_x, acc_y), label=f"Acc for Repeat {i+1}")
        plt.legend()
        plt.ylabel("Acceleration (m/s^2)")
        plt.xlabel("Time (s)")

         # temporary save
        out_path = os.path.join('./', f"repeat_localization_error_{i+1}.npz")
        np.savez_compressed(out_path, x=x, y=y, t=t, dist=dist)

    plt.show()

