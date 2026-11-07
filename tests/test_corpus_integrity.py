import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.build_corpus_manifest import verified_bin
from src.train_mix_v3 import committed_bins


class CorpusIntegrityTests(unittest.TestCase):
    def test_verified_bin_rejects_path_escape_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domain = root / "v3_code"
            domain.mkdir()
            payload = domain / "chunk.bin"
            payload.write_bytes(b"1234")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()

            self.assertEqual(verified_bin(root, "v3_code", "chunk.bin", 4, digest), payload.resolve())
            self.assertIsNone(verified_bin(root, "v3_code", "../chunk.bin", 4, digest))
            self.assertIsNone(verified_bin(root, "v3_code", "chunk.bin", 4, "0" * 64))

    def test_committed_bins_rejects_same_size_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domain = root / "v3_code"
            domain.mkdir()
            payload = domain / "chunk.bin"
            payload.write_bytes(b"1234")
            (domain / "committed_chunks.jsonl").write_text(
                json.dumps({"bin": "chunk.bin", "tokens": 1,
                             "training_approved": True, "sha256": "0" * 64}) + "\n"
            )

            self.assertEqual(committed_bins(root), {})


if __name__ == "__main__":
    unittest.main()
