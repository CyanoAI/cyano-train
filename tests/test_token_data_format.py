import struct
import tempfile
import unittest
from pathlib import Path

from src.tok_data import write_tokens


class TokenDataFormatTests(unittest.TestCase):
    def test_writes_little_endian_uint32_for_training_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.bin"

            written = write_tokens(path, [1, 2, 50256, 4, 5], seq_len=4)

            self.assertEqual(written, 4)
            self.assertEqual(path.stat().st_size, 4 * 4)
            self.assertEqual(struct.unpack("<4I", path.read_bytes()), (1, 2, 50256, 4))


if __name__ == "__main__":
    unittest.main()
