import numpy as np
from pylgmath.se3.transformation import Transformation


from vtr_pose_graph.graph import Graph, GraphError
from vtr_pose_graph.graph_iterators import DepthFirstSearchIterator, SpatialIterator, TemporalIterator, BreadthFirstSearchIterator
from vtr_pose_graph.vertex import Vertex


def get_closest_teach_vertex(v: Vertex) -> Vertex:
    if v.taught:
        return v
    for r_v, _ in BreadthFirstSearchIterator(v):
        if r_v.taught:
            return r_v
    raise GraphError("Graph is malformed, repeat pass does not connect to teach vertex.")

def collect_path(v_start: Vertex, v_goal: Vertex):

    def _collect_path(it, v_goal: Vertex):
        p = []
        for v, e in it:
            p.append(e)
            if v == v_goal:
                return p
        raise GraphError(f"{v_goal} was not connected to the iterator")

    if v_start == v_goal:
        return []
    path = []
    if v_start.run != 0:
        start_teach_v = get_closest_teach_vertex(v_start)
        path += _collect_path(SpatialIterator(v_start), start_teach_v)
        v_start = start_teach_v
    if v_goal.run != 0:
        goal_teach_v = get_closest_teach_vertex(v_goal)
        path += _collect_path(TemporalIterator(v_start), goal_teach_v)
        path += _collect_path(SpatialIterator(goal_teach_v, to_teach=False), v_goal)
    else:
        path += _collect_path(TemporalIterator(v_start), v_goal)
    return path
   
def count_runs(graph: Graph) -> int :
    max_run = 0
    for r_v, _ in BreadthFirstSearchIterator(graph.root):
        max_run = max(r_v.run, max_run)
    return max_run + 1

def set_world_frame(graph: Graph, vertex: Vertex, T_w_v0=Transformation()):
    vertex.T_w_v = T_w_v0

    for v, e in DepthFirstSearchIterator(vertex):
        if v == vertex:
            continue
        
        v_from = graph.get_vertex(e.from_id)
        v_to = graph.get_vertex(e.to_id)

        v.T_w_v = v_to.T_w_v @ e.T if v == v_from else v_from.T_w_v @ e.T.inverse()
        #print(v.T_w_v)

def mask_points_near_vertex(vertex: Vertex, points:np.ndarray, distance: float):
    return np.linalg.norm(points[:, :2] - vertex.T_v_w.r_ba_ina()[:2].reshape(1, 2), axis=1) < distance

def filter_points_along_path(root_v: Vertex, points: np.ndarray, distance: float):
    mask = np.zeros((points.shape[0],), dtype=bool)
    for v, _ in TemporalIterator(root_v):
        mask |= mask_points_near_vertex(v, points, distance)
    return mask



