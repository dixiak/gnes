import unittest
import numpy as np
import ctypes
from src.nes.indexer.findexer import FIndexer


class TestFIndexer(unittest.TestCase):
    def setUp(self):
        self.n_bytes = 20
        self.n_lines = 100000
        self.top_k = 2
        self.query_num = 100
        self.test_ints = np.random.randint(
                        0, 255, [self.n_lines, self.n_bytes]).astype(np.uint8)

        self.test_bytes = self.test_ints.tobytes()

        self.test_docids = [np.random.randint(0, ctypes.c_uint(-1).value)
                            for _ in range(self.n_lines)]

        self.query_bytes = self.test_ints[:self.query_num].tobytes()
        self.query_result = []

        for i in range(self.query_num):
            rk = np.sum(np.minimum(np.abs(
                        self.test_ints - self.test_ints[i]), 1), -1)
            rk = sorted(enumerate(rk), key=lambda x: x[1])[:self.top_k]
            self.query_result.append([(self.test_docids[k], r/self.n_bytes)
                                      for k, r in rk])

    def test_add(self):
        fd = FIndexer()
        fd.add(self.test_bytes, self.test_docids)
        self.assertEqual(self.n_bytes, fd.num_bytes)
        self.assertEqual(self.n_lines, len(fd.doc_ids))
        self.assertEqual(self.n_lines, len(fd.vectors))

    def test_query(self):
        fd = FIndexer()
        fd.add(self.test_bytes, self.test_docids)
        res = fd.query(self.query_bytes, self.top_k)

        rt = sum([self.query_result[i][j][0] == res[i][j][0]
                 for i in range(self.query_num)
                 for j in range(self.top_k)])
        rt2 = sum([self.query_result[i][j][1] == res[i][j][1]
                  for i in range(self.query_num)
                  for j in range(self.top_k)])

        self.assertEqual(rt, self.top_k * self.query_num)
        self.assertEqual(rt2, self.top_k * self.query_num)
