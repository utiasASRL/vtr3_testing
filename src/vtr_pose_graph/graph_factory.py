import abc

from vtr_pose_graph.graph import Graph


class GraphFactory(abc.ABC):

    @abc.abstractmethod
    def buildGraph(self) -> Graph:
        return Graph()
