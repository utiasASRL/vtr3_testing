import pytest

import os
import matplotlib.pyplot as plt
import numpy as np
from vtr_regression_testing.odometry_chain import OdometryChainFactory

from vtr_pose_graph.graph_iterators import TemporalIterator
import argparse



@pytest.fixture
def graph_path(request):
    candidate_graph_dir = os.path.join(os.getenv("VTRTEMP"), request.config.getoption('--graph'))
    return candidate_graph_dir

@pytest.fixture
def baseline_path(request):
    baseline_graph_dir = os.path.join(os.getenv("VTRROOT"), "vtr_testing", "baseline", request.config.getoption('--baseline'))
    return baseline_graph_dir

@pytest.fixture
def graph(graph_path):
    factory = OdometryChainFactory(graph_path)
    return factory.buildGraph()

@pytest.fixture
def baseline(baseline_path):
    factory = OdometryChainFactory(baseline_path)
    return factory.buildGraph()

def test_graph_sizes(graph, baseline):
    assert graph.number_of_edges == baseline.number_of_edges, "Graphs have unequal number of edges"
    assert graph.number_of_vertices == baseline.number_of_vertices, "Graphs have unequal number of vertices"

def test_pose_error(graph, baseline):
    pose_err = []
    ang_err = []
    t = []

    for (v1, _), (v2, _) in zip(TemporalIterator(baseline.root), TemporalIterator(graph.root)):
        pose_err.append(v1.T_v_w.r_ba_ina()-v2.T_v_w.r_ba_ina())
        ang_err.append(v1.T_v_w.C_ba() @ v2.T_w_v.C_ba())
        t.append(v1.stamp / 1e9)

    sq_pose_err = np.linalg.norm(np.array(pose_err), axis=0)**2
    sq_angle_err = np.array([np.arccos((np.trace(C) - 1) / 2) for C in ang_err])**2

    pose_rms = np.sqrt(np.mean(sq_pose_err))
    ang_rms = np.rad2deg(np.sqrt(np.nanmean(sq_angle_err)))

    print(f"RMS Position {pose_rms}m")
    print(f"RMS Angle {ang_rms}deg")

    assert pose_rms < 0.001, "RMS error should be less than 1mm for the same dataset"
    assert ang_rms < 0.1, "RMS error should be less than 0.1 degree for the same dataset"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Regression Test Odometry',
                        description = 'Verify that two pose graphs are the same, within a small tolerance')
    parser.add_argument('-g', '--graph', default="graph")      # option that takes a value
    parser.add_argument('-b', '--baseline', default="baseline_graph")      # option that takes a value
    args = parser.parse_args()

    baseline_graph_dir = os.path.join(os.getenv("VTRROOT"), "vtr_testing", "baseline", args.baseline)
    candidate_graph_dir = os.path.join(os.getenv("VTRTEMP"), args.graph)

    base_factory = OdometryChainFactory(baseline_graph_dir)
    baseline_graph = base_factory.buildGraph()

    can_factory = OdometryChainFactory(candidate_graph_dir)
    candidate_graph = can_factory.buildGraph()

    assert baseline_graph.number_of_edges == candidate_graph.number_of_edges, "Graphs have unequal number of edges"
    assert baseline_graph.number_of_vertices == candidate_graph.number_of_vertices, "Graphs have unequal number of vertices"


    print(f"Baseline Graph {baseline_graph} has {baseline_graph.number_of_vertices} vertices and {baseline_graph.number_of_edges} edges")
    print(f"Candidate Graph {candidate_graph} has {candidate_graph.number_of_vertices} vertices and {candidate_graph.number_of_edges} edges")

    pose_err = []
    ang_err = []
    t = []

    for (v1, _), (v2, _) in zip(TemporalIterator(baseline_graph.root), TemporalIterator(candidate_graph.root)):
        pose_err.append(v1.T_v_w.r_ba_ina()-v2.T_v_w.r_ba_ina())
        ang_err.append(v1.T_v_w.C_ba() @ v2.T_w_v.C_ba())
        t.append(v1.stamp / 1e9)

    sq_pose_err = np.linalg.norm(np.array(pose_err), axis=0)**2
    sq_angle_err = np.array([np.arccos((np.trace(C) - 1) / 2) for C in ang_err])**2

    print(f"RMS Position {np.sqrt(np.mean(sq_pose_err))}m")
    print(f"RMS Angle {np.rad2deg(np.sqrt(np.nanmean(sq_angle_err)))}deg")
        

    plt.plot(t, sq_angle_err, label="Position Error")
    plt.xlabel("Time (s)")
    plt.show()

