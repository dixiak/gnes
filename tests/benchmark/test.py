from gnes.client.base import ZmqClient
from gnes.cli.parser import set_service_parser
from gnes.service.base import SocketType
from gnes.proto import gnes_pb2
from gnes.proto import array2blob, blob2array
import numpy as np


port_a = 1121
port_b = 1122


class mock_input():
    def __init__(self):
        pass

    @staticmethod
    def video_encode_input(num_docs=2,
                           num_shots=2,
                           num_pics=2,
                           input_width=5,
                           input_height=5,
                           input_channel=3):
        return [np.random.randint(0, 255, [
                num_shots, num_pics, input_width, input_height, input_channel]).astype(np.uint8)
                for _ in range(num_docs)]

    @staticmethod
    def float_index_input(num_docs=2,
                          num_shots=10,
                          num_dim=300):
        return [np.random.random([num_shots, num_dim]) for _ in range(num_docs)]


class mock_call():
    def __init__(self, port_in=1121,
                 port_out=1122,
                 socket_in=SocketType.PULL_CONNECT,
                 socket_out=SocketType.PUSH_CONNECT):
        self.port_in = str(port_in)
        self.port_out = str(port_out)
        self.socket_in = str(socket_in)
        self.socket_out = str(socket_out)
        args = set_service_parser().parse_args(
            ['--port_in', self.port_in,
             '--port_out', self.port_out,
             '--socket_in', self.socket_in,
             '--socket_out', self.socket_out])
        self.client = ZmqClient(args)

    def _fake_encode_msg(self, input_list,
                         defined_type=gnes_pb2.Document.VIDEO):
        msg = gnes_pb2.Message()
        req = gnes_pb2.Request()

        for doc in input_list:
            d = req.index.docs.add()
            d.doc_type = defined_type
            #d.doc_type=3
            for _ in range(len(doc)):
                c = d.chunks.add()
                c.blob.CopyFrom(array2blob(doc[_]))
        msg.request.CopyFrom(req)

        return msg

    def _fake_index_msg(self,
                        input_list):
        msg = gnes_pb2.Message()

        for doc in input_list:
            d = msg.request.index.docs.add()
            d.doc_id = np.random.randint(0, 1e9)
            for idx, vec in enumerate(doc):
                c = d.chunks.add()
                c.embedding.CopyFrom(array2blob(vec))
                c.offset = idx
                c.weight = 1

        return msg

    def mock_encode_call(self, input_list):

        msg = self._fake_encode_msg(input_list)
        self.client.send_message(msg)
        r = self.client.recv_message(timeout=-1)
        embeds = []
        for d in r.request.index.docs:
            embeds.append([blob2array(c.embedding) for c in d.chunks])

        return embeds

    def mock_index_add(self, input_list):
        msg = self._fake_index_msg(input_list)
        self.client.send_message(msg)
        r = self.client.recv_message(timeout=-1)
        if r:
            return 'suc'

    def mock_index_query(self, input_single):
        msg = gnes_pb2.Message()
        for _ in input_single:
            c = msg.request.search.query.chunks.add()
            c.embedding.CopyFrom(array2blob(_))
        self.client.send_message(msg)
        return self.client.recv_message(timeout=-1)


doc_list = mock_input.float_index_input()
c = mock_call()
print(c.mock_index_add(doc_list))

'''
doc_list = mock_input.video_encode_input(10, 2, 2, 5, 5, 3)
c = mock_call()
c.mock_encode_call(doc_list)
'''