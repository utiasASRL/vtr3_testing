import numpy as np
from vtr_pose_graph import INVALID_ID, Vertex, Edge
from vtr_pose_graph.edge import EDGE_TYPE_TEMPORAL
from vtr_pose_graph.graph import Graph
from vtr_pose_graph.graph_factory import GraphFactory
from vtr_utils.bag_file_parsing import BagFileCache, format_rosbag_name
from pylgmath.se3.transformation import Transformation



class OdometryChainFactory (GraphFactory):
    """This class is a special graph with only 1 run, that can be used for odometry evaluation.
        Because many of these vertices do not have data associated with them, data access is prevented
    """

    def __init__(self, rosbag2_root):
        self.odom_path = format_rosbag_name("data", "odometry_result")
        self.cache = BagFileCache(rosbag2_root)
        self.root = rosbag2_root

    
    def buildGraph(self) -> Graph:
        g = Graph()
        g.major_id = 0

        odo_access = self.cache[self.odom_path]

        last_vid = INVALID_ID

        for _, odom_msg in odo_access.get_bag_msgs_iter("odometry_result"):
            odo_vertex = Vertex()
            odo_vertex.stamp = odom_msg.timestamp
            odo_vertex.T_w_v = Transformation(xi_ab=np.array(odom_msg.t_world_robot.xi).reshape(6, 1))

            if (last_vid == INVALID_ID):
                odo_vertex.id = 0
                g.add_vertex(odo_vertex)
            else:
                odo_vertex.id = last_vid + 1
                g.add_vertex(odo_vertex)

                edge = Edge()
                edge.from_id = last_vid
                edge.to_id = odo_vertex.id
                edge.T = g.get_vertex(last_vid).T_v_w * odo_vertex.T_w_v
                edge.type = EDGE_TYPE_TEMPORAL

                g.add_edge(edge)
            
            last_vid = odo_vertex.id
        
        g.minor_id = g.number_of_vertices

        return g





