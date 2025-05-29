import numpy as np
from vtr_pose_graph import INVALID_ID, Vertex, Edge
from vtr_pose_graph.edge import EDGE_TYPE_SPATIAL
from vtr_pose_graph.graph import Graph
from vtr_pose_graph.graph_factory import GraphFactory
from vtr_pose_graph.graph_iterators import BreadthFirstSearchIterator, spatial_filter
from vtr_pose_graph.vertex import Vertex
from vtr_utils.bag_file_parsing import BagFileCache, format_rosbag_name
from pylgmath.se3.transformation import Transformation


class LocalizationIterator (BreadthFirstSearchIterator):
    def __init__(self, vertex_start: Vertex):
        super().__init__(vertex_start, spatial_filter)


class LocalizationChainFactory(GraphFactory):

    def __init__(self, rosbag2_root):
        self.idx_path = format_rosbag_name("", "index")
        self.edge_path = format_rosbag_name("", "edges")
        self.vertex_path = format_rosbag_name("", "vertices")        
        self.loc_path = format_rosbag_name("data", "localization_result")
        self.cache = BagFileCache(rosbag2_root)

    def buildGraph(self) -> Graph:
        root_access = self.cache[self.idx_path]
        index_msgs = root_access.get_bag_messages("index")

        assert len(index_msgs) == 1, "Index can have only 1 message, investigate your file paths or " \
                "vtr_pose_graph structure has changed"
        graph = Graph(index_msgs[0][1])

        vert_access = self.cache[self.vertex_path]
        for _, vert_msg in vert_access.get_bag_msgs_iter("vertices"):
            graph.add_vertex(Vertex(vert_msg, bagfile_cache=self.cache))

        edge_access = self.cache[self.edge_path]
        for _, edge_msg in edge_access.get_bag_msgs_iter("edges"):
            edge = Edge(edge_msg)

            #For localization we want only teach paths
            if edge.is_teach():
                graph.add_edge(edge)
        
        loc_access = self.cache[self.loc_path]
        for vid, (_, loc_msg) in enumerate(loc_access.get_bag_msgs_iter("localization_result")):
            loc_vertex = Vertex()
            loc_vertex.id = Vertex.compose_id(graph.major_id + 1, vid)
            loc_vertex.stamp = loc_msg.timestamp
            graph.add_vertex(loc_vertex)

            loc_edge = Edge()
            loc_edge.from_id = loc_vertex.id
            loc_edge.to_id = loc_msg.vertex_id
            loc_edge.T = Transformation(xi_ab=np.array(loc_msg.t_robot_vertex.xi).reshape(6, 1))
            loc_edge.type = EDGE_TYPE_SPATIAL

            graph.add_edge(loc_edge)


        return graph


