import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from src.cyano_train import checkpoint_space_available


class CheckpointSpaceTests(unittest.TestCase):
    def test_distributed_decision_uses_minimum_available_space(self):
        def emulate_minimum(value, op):
            self.assertEqual(op, dist.ReduceOp.MIN)
            value.fill_(1.5)

        with patch("src.cyano_train.dist.all_reduce", side_effect=emulate_minimum) as reduce:
            can_save, free_gb = checkpoint_space_available(8.0, 2.0, 2, torch.device("cpu"))

        self.assertFalse(can_save)
        self.assertEqual(free_gb, 1.5)
        reduce.assert_called_once()

    def test_single_rank_uses_local_available_space(self):
        with patch("src.cyano_train.dist.all_reduce") as reduce:
            can_save, free_gb = checkpoint_space_available(3.0, 2.0, 1, torch.device("cpu"))

        self.assertTrue(can_save)
        self.assertEqual(free_gb, 3.0)
        reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main()
