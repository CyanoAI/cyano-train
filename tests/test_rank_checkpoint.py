import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.rank_checkpoint import load_rank_checkpoint, save_rank_checkpoint


class RankCheckpointTests(unittest.TestCase):
    def test_commits_two_slots_and_loads_latest_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "checkpoint.pt")

            save_rank_checkpoint(path, {"step": 3, "value": torch.tensor([3])}, 0, 1)
            first = load_rank_checkpoint(path, 0, 1)
            self.assertEqual(first["step"], 3)
            self.assertEqual(first["value"].item(), 3)
            self.assertNotIn("legacy_resume", first)
            self.assertEqual(json.loads(Path(path + ".manifest.json").read_text())["slot"], 0)

            save_rank_checkpoint(path, {"step": 4, "value": torch.tensor([4])}, 0, 1)
            latest = load_rank_checkpoint(path, 0, 1)
            self.assertEqual(latest["step"], 4)
            self.assertEqual(latest["value"].item(), 4)
            self.assertEqual(json.loads(Path(path + ".manifest.json").read_text())["slot"], 1)

    def test_rejects_world_mismatch_without_changing_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "checkpoint.pt")
            save_rank_checkpoint(path, {"step": 3}, 0, 1)
            manifest_path = Path(path + ".manifest.json")
            before = manifest_path.read_bytes()

            with self.assertRaises(ValueError):
                load_rank_checkpoint(path, 0, 2)

            with self.assertRaises(RuntimeError):
                save_rank_checkpoint(path, {"step": 4}, 0, 2)
            self.assertEqual(manifest_path.read_bytes(), before)

    def test_loads_legacy_checkpoint_as_inexact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"step": 8, "value": 12}, path)

            state = load_rank_checkpoint(str(path), 0, 1)

            self.assertEqual(state["step"], 8)
            self.assertTrue(state["legacy_resume"])


if __name__ == "__main__":
    unittest.main()
