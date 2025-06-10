import numpy as np
import pytest

from vtr_regression_testing.path_comparison import distance_to_path, distances_to_path

@pytest.fixture
def short_path():
    a = 1/np.sqrt(2)
    return np.array([[0, 0, 0, 1, 0, 0, 2],
                     [2, 0, 0, a, a, 0, np.sqrt(2)],
                     [3, 1, 0, -a, -a, 0, np.sqrt(2)]])


@pytest.mark.parametrize("point,expected, message", [(np.array([0.5, 0, 0]), 0, "Intermediate on path"),
                                            (np.array([0.25, 0.5, 0]), 0.5, "Intermediate off of path projection"),
                                            (np.array([3.0, 0, 0]), 1/np.sqrt(2), "Error if paths are infinite, projection"),
                                            (np.array([3.5, 1.0, 0]), 0.5, "Past the end of the path, raw keypoint distance used"),
                                            (np.array([-3.0, 0.0, 0]), 3.0, "Past the start of the path, raw keypoint distance used")])
def test_distance_to_path(short_path, point, expected, message):
    assert distance_to_path(point, short_path) == expected, message


@pytest.mark.parametrize("points,expected, message", [(np.array([[0.5, 0, 0], [0.25, 1, 0], [0.25, 0.5, 0], [3.0, 0, 0], [3.5, 1.0, 0], [-3.0, 0.0, 0]]), 
                                                       np.array([0, 1, 0.5, 1/np.sqrt(2), 0.5, 3.0]), "Check all simultaneously"),])
def test_distances_to_path(short_path, points, expected, message):
    assert (distances_to_path(points, short_path) == expected).all(), message

