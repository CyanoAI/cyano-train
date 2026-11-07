import unittest

import numpy as np

from src.train_mix_v3 import EOS, split_docs


class SplitDocsTests(unittest.TestCase):
    def test_keeps_unterminated_final_document(self):
        docs = split_docs(np.array([1, EOS, 2, 3], dtype=np.uint32))

        self.assertEqual([doc.tolist() for doc in docs], [[1, EOS], [2, 3]])

    def test_keeps_document_without_eos(self):
        docs = split_docs(np.array([1, 2, 3], dtype=np.uint32))

        self.assertEqual([doc.tolist() for doc in docs], [[1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
