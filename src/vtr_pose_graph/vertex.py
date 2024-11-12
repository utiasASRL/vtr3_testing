import numpy as np
from pylgmath.se3.transformation import Transformation


from vtr_pose_graph import INVALID_ID
from vtr_pose_graph.edge import Edge


class Vertex:

    def __init__(self, vertex_msg=None, bagfile_cache=None):
        if vertex_msg:
            self.id = vertex_msg.id
            self.stamp = vertex_msg.vertex_time.nanoseconds_since_epoch
        else:
            self.id = INVALID_ID
            self.stamp = 0
        
        self.T_w_v = Transformation()
        self.cache = bagfile_cache
        self.neighbours = set()
        self.diff = None
        self.taught = False

    @property
    def T_v_w(self):
        return self.T_w_v.inverse()
    
    @property
    def run(self):
        return self.id >> 32
    
    @property
    def minor_id(self):
        return self.id & 0x00000000FFFFFFFF;

    def get_data(self, message_name: str):
        if not self.cache:
            raise IndexError("Vertex does not have access to any data")
        db = self.cache.get_data(message_name)
        return db.get_bag_message(message_name, self.stamp)

    def write_data(self, message_name: str, data):
        if not self.cache:
            raise IndexError("Vertex does not have access to any data")
        db = self.cache.get_data(message_name)
        raise NotImplementedError("Writing to pose graphs is not yet supported.")

    def get_neighbours(self):
        return sorted(self.neighbours, key=lambda t:t[1].id, reverse=True)

    def link(self, v, e):
        self.neighbours.add((v, e))
        self._mark_teach(e)

    def _mark_teach(self, e: Edge):
        if e.is_teach() and self.id == e.to_id:
            self.taught = True


    @staticmethod
    def compose_id(run_id, vertex_id):
        return (run_id << 32) | vertex_id


    def __repr__(self):
        return f"Vertex <{self.run},{self.minor_id}>"