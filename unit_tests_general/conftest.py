import pytest

def pytest_addoption(parser):
    parser.addoption(
        '--graph', action='store', default='lidar/graph', help='Graph under test'
    )
    parser.addoption(
        '--baseline', action='store', default='honeycomb_flat_loop', help='Baseline graph for comparison'
    )