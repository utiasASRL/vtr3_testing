from __future__ import annotations

import typing

import numpy as np
if  typing.TYPE_CHECKING:
    from vtr_pose_graph import GraphIterator
    from vtr_pose_graph.graph import Graph
from vtr_pose_graph import INVALID_ID


def path_to_matrix(graph: Graph, path: GraphIterator,transform = None):
    """The vertices of the graph are assumed to be defined in the world frame."""
    
    #First 3 are the position of the start of the line, second three are the unit vector along the path segment
    points = np.zeros((0, 7))
    for v, e in path:
        if e.from_id == INVALID_ID or e.to_id == INVALID_ID:
            continue
        x0 = graph.get_vertex(e.from_id).T_v_w.r_ba_ina()
        x1 = graph.get_vertex(e.to_id).T_v_w.r_ba_ina()
        if transform is not None:
            x0 = (transform @ graph.get_vertex(e.from_id).T_v_w).r_ba_ina()
            x1 = (transform @ graph.get_vertex(e.to_id).T_v_w).r_ba_ina()
        
        l = np.linalg.norm(x1 - x0)
        n = (x1 - x0) / l
        row = np.zeros((1, 7))
        row[0, :3] = x0.T
        row[0, 3:6] = n.T
        row[0, 6] = l
        points = np.vstack((points, row))
    if points.shape[0] > 0:
        final_row = np.zeros((1, 7))
        final_row[0, :3] = x1.T
        final_row[0, 3:6] = -n.T
        final_row[0, 6] = l
        points = np.vstack((points, final_row))

    return points

def distance_to_path(point: np.ndarray, path: np.ndarray):
    """Path is assumed to be formatted from path_to_matrix"""

    assert len(path.shape) == 2 and path.shape[1] == 7, "Matrix is ill formed. Should be nx7"
    assert len(point.shape) <= 2 and point.size == 3, "Point must be 3 dimensional"
    
    x0 = path[:, :3]
    n = path[:, 3:6]
    l = path[:, 6]
    points = np.zeros_like(x0)
    points[:] = point.T

    norm_dist = np.linalg.norm(np.cross((points - x0), n), axis=1)
    dists = np.linalg.norm((points - x0), axis=1)
    min_dist = np.min(dists)

    proj = np.diagonal(np.dot(points - x0, n.T))
    segment_filter = (proj > 0) & (proj < l)

    min_proj = 1e6

    #Handle the edge cases where points are beyond the limits of the path
    if np.sum(segment_filter) > 0:
        min_proj = np.min(norm_dist[segment_filter])
    
    return min(min_proj, min_dist)

def signed_distance_to_path(point: np.ndarray, path: np.ndarray):
    """Path is assumed to be formatted from path_to_matrix"""

    assert len(path.shape) == 2 and path.shape[1] == 7, "Matrix is ill formed. Should be nx7"
    assert len(point.shape) <= 2 and point.size == 3, "Point must be 3 dimensional"
    
    x0 = path[:, :3]
    tangent = path[:, 3:6]
    l = path[:, 6]
    points = np.zeros_like(x0)
    points[:] = point.T

    norm_vec = np.cross((points - x0), tangent)
    norm_dist = np.linalg.norm(norm_vec, axis=1)
    dists = np.linalg.norm((points - x0), axis=1)
    min_dist = np.min(dists)

    proj = np.diagonal(np.dot(points - x0, tangent.T))
    segment_filter = (proj > 0) & (proj < l)

    min_proj = 1e6

    #Handle the edge cases where points are beyond the limits of the path
    if np.sum(segment_filter) > 0:
        idx_min_proj = np.argmin(norm_dist[segment_filter])
        if norm_dist[segment_filter][idx_min_proj] < min_dist:
            return np.sign(norm_vec[segment_filter][idx_min_proj, 2]) * norm_dist[segment_filter][idx_min_proj]
    return min_dist

def distances_to_path(points: np.ndarray, path: np.ndarray):
    """Path is assumed to be formatted from path_to_matrix"""

    assert len(path.shape) == 2 and path.shape[1] == 7, "Matrix is ill formed. Should be nx7"
    assert len(points.shape) == 2 and points.shape[1] == 3, "Points are ill formed. Should be mx3"
    n_path = path.shape[0]
    m_points = points.shape[0]

    batched_path = np.broadcast_to(path[..., None],path.shape+(m_points,)) # N = m here
    
    x0 = batched_path[:, :3, :]
    n = batched_path[:, 3:6, :]
    l = batched_path[:, 6, :]
    batched_points = np.broadcast_to(points.T, (n_path, )+points.T.shape)

    
    norm_dist = np.linalg.norm(np.cross((batched_points - x0), n, axis=1), axis=1)
    dists = np.linalg.norm((batched_points - x0), axis=1)
    min_dist = np.min(dists, axis=0)

    assert min_dist.size == m_points, "min dist should be for each point"

    proj = np.einsum('ijk,ijk->ik', batched_points - x0, n)
    segment_filter = (proj > 0) & (proj < l)

    min_proj = 1e6

    #Handle the edge cases where points are beyond the limits of the path
    if np.sum(segment_filter) > 0:
        masked_norms = np.ma.array(norm_dist, mask=~segment_filter)
        min_proj = np.min(masked_norms, axis=0)
    
    return np.min(np.vstack([min_proj, min_dist]), axis=0).filled(0)

