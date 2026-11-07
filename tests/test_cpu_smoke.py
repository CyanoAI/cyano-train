import json
import os
import struct
import textwrap
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CpuSmokeTests(unittest.TestCase):
    def test_train_save_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            data = work / "tokens.bin"
            tokens = [i % 128 for i in range(512)]
            data.write_bytes(struct.pack(f"<{len(tokens)}I", *tokens))
            checkpoint = work / "checkpoint.pt"
            logs = work / "logs"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src")
            env["CUDA_VISIBLE_DEVICES"] = ""

            def run(*args):
                return subprocess.run(
                    [sys.executable, str(ROOT / "src" / "cyano_train.py"),
                     "--rank", "0", "--world", "1", "--steps", "2",
                     "--max_run_steps", "1", "--batch_size", "1",
                     "--seq_len", "8", "--warmup", "1", "--eval_every", "1",
                     "--save_every", "0", "--optimizer", "muon",
                     "--muon_ns_steps", "1", "--config", "smoke",
                     "--data", str(data), "--logdir", str(logs),
                     "--ckpt", str(checkpoint), "--tag", "cpu", *args],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
                )

            first = run()
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertIn('"last_step": 1', first.stdout)
            self.assertIn('"skips": 0', first.stdout)
            self.assertTrue(Path(str(checkpoint) + ".manifest.json").exists())

            second = run("--resume")
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("RESUMED from step 1", second.stdout)
            self.assertIn('"last_step": 2', second.stdout)
            self.assertIn('"eval_loss"', (logs / "rank0_cpu.jsonl").read_text())

            records = [json.loads(line) for line in
                       (logs / "rank0_cpu.jsonl").read_text().splitlines()
                       if line.startswith("{")]
            self.assertEqual(sum(record.get("event") == "startup" for record in records), 2)

    def test_checkpoint_space_skip_is_not_reported_as_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            data = work / "tokens.bin"
            tokens = [i % 128 for i in range(512)]
            data.write_bytes(struct.pack(f"<{len(tokens)}I", *tokens))
            checkpoint = work / "checkpoint.pt"
            logs = work / "logs"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src")
            env["CUDA_VISIBLE_DEVICES"] = ""
            runner = textwrap.dedent(f"""
                import sys
                import torch
                import cyano_train

                torch.cuda.is_available = lambda: False
                cyano_train.checkpoint_space_available = lambda *args: (False, 0.0)
                sys.argv = [
                    "cyano_train.py", "--rank", "0", "--world", "1",
                    "--steps", "1", "--max_run_steps", "1",
                    "--batch_size", "1", "--seq_len", "8", "--warmup", "1",
                    "--eval_every", "0", "--optimizer", "muon",
                    "--muon_ns_steps", "1", "--config", "smoke",
                    "--data", {str(data)!r}, "--logdir", {str(logs)!r},
                    "--ckpt", {str(checkpoint)!r},
                ]
                cyano_train.main()
            """)
            result = subprocess.run(
                [sys.executable, "-c", runner], cwd=ROOT, env=env,
                capture_output=True, text=True, timeout=120,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn('"status": "checkpoint_skipped"', result.stdout)
            self.assertIn("without a committed checkpoint", result.stderr)
            self.assertFalse(Path(str(checkpoint) + ".manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
