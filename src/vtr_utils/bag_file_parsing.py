import os
import sqlite3

from vtr_pose_graph.edge import Edge
from vtr_pose_graph.graph import Graph
from vtr_pose_graph.graph_factory import GraphFactory
from vtr_pose_graph.vertex import Vertex

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def format_rosbag_name(root_dir, bagfile_name):
    """The vtr pose graph convention is that data is stored like graph/data/topic/topic_0.db3"""
    return os.path.join(root_dir, bagfile_name, f"{bagfile_name}_0.db3")


class BagFileParser():
    """The BagFileParser assumes that the database is formatted with cdr compression, the standard for ros2 prior to 
    MCAP in ROS2 Iron. vtr3 still uses cdr at this time."""

    def __init__(self, bag_file):
        try:
            self.conn = sqlite3.connect(bag_file)
        except Exception as e:
            print('Could not connect: ', e)
            raise Exception(f'could not connect to {bag_file}')

        self.cursor = self.conn.cursor()

        ## create a message (id, topic, type) map
        topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        self.topic_id = {name_of: id_of for id_of, name_of, type_of in topics_data}
        self.topic_msg_message = {name_of: get_message(type_of) for id_of, name_of, type_of in topics_data}

    def get_bag_message(self, topic_name, time_stamp):
        topic_id = self.topic_id[topic_name]
        rows = self.cursor.execute(
            "SELECT data FROM messages WHERE topic_id = {} AND timestamp ={}".format(topic_id, time_stamp)).fetchall()
        
        assert len(rows) > 0, f"Error, no message at timestamp {time_stamp} for type {topic_name}"

        assert len(rows) == 1, "Error in database, vertex has multiple matching messages"
        return deserialize_message(rows[0][0], self.topic_msg_message[topic_name])

    def write_bag_mesage(self, topic_name, timestamp, msg):
        raise NotImplementedError("Currently, only reading from a pose graph is allowed")

    # Return messages as list of tuples [(timestamp0, message0), (timestamp1, message1), ...]
    def get_bag_messages(self, topic_name):
        topic_id = self.topic_id[topic_name]
        rows = self.cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
        return [(timestamp, deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp, data in rows]

    def get_bag_msgs_iter(self, topic_name):
        topic_id = self.topic_id[topic_name]
        result = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {} ORDER BY id".format(topic_id))
        while True:
            res = result.fetchone()
            if res is not None:
                yield (res[0], deserialize_message(res[1], self.topic_msg_message[topic_name]))
            else:
                break


class BagFileCache:
    """The bagfile cache is a simple wrapper that allows all vertices access to the same
    database connection to read info. This greatly speeds up access, but is not thread safe."""

    def __init__(self, graph_root):
        self.graph_root = graph_root
        self.dbs = {}

    def get_data(self, topic):
        return self[format_rosbag_name("data", topic)]

    def __getitem__(self, db_name):
        if not db_name in self.dbs.keys():
            self.dbs[db_name] = BagFileParser(os.path.join(self.graph_root, db_name))
        return self.dbs[db_name]

    def has_topic(self, topic):
        return topic in os.listdir(os.path.join(self.graph_root, 'data'))


class Rosbag2GraphFactory(GraphFactory):

    def __init__(self, rosbag2_root):
        self.idx_path = format_rosbag_name("", "index")
        self.edge_path = format_rosbag_name("", "edges")
        self.vertex_path = format_rosbag_name("", "vertices")
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
            graph.add_edge(Edge(edge_msg))

        return graph