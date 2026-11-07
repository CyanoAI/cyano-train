import json
import time
import struct
import argparse
from pathlib import Path


def write_tokens(output_path, tokens, seq_len):
    total_tokens = len(tokens) // seq_len * seq_len
    with open(output_path, "wb") as output:
        for start in range(0, total_tokens, seq_len):
            chunk = tokens[start:start + seq_len]
            output.write(struct.pack(f"<{len(chunk)}I", *chunk))
    return total_tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-bytes", type=int, default=200 * 1024 * 1024)
    a = ap.parse_args()

    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(a.tokenizer)
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    seq_len = a.seq_len
    max_bytes = a.max_bytes

    input_dir = Path(a.input_dir)
    output_path = Path(a.output)

    priority_files = [
        "belle_train_2m_cn.jsonl",
        "wikipedia_zh.jsonl",
        "camel_math.jsonl",
        "camel_biology.jsonl",
        "camel_chemistry.jsonl",
        "camel_physics.jsonl",
    ]

    all_tokens = []
    total_bytes_read = 0
    total_lines = 0
    files_processed = []
    t0 = time.time()

    for fname in priority_files:
        fpath = input_dir / fname
        if not fpath.exists():
            print(f"Skip missing: {fname}")
            continue
        fsize = fpath.stat().st_size
        print(f"Processing: {fname} ({fsize/1e6:.1f}MB on disk, already read {total_bytes_read/1e6:.1f}MB text)")
        line_count = 0
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if total_bytes_read >= max_bytes:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                except json.JSONDecodeError:
                    continue
                if not text:
                    continue
                total_bytes_read += len(text.encode("utf-8"))
                ids = tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(ids)
                line_count += 1
                total_lines += 1
                if total_lines % 50000 == 0:
                    elapsed = time.time() - t0
                    print(f"  {total_lines} lines, {total_bytes_read/1e6:.1f}MB text, {len(all_tokens)} tokens, {elapsed:.1f}s, {len(all_tokens)/elapsed:.0f} tok/s")
        files_processed.append((fname, line_count))
        if total_bytes_read >= max_bytes:
            print(f"Reached {max_bytes/1e6:.0f}MB text limit, stopping")
            break

    elapsed = time.time() - t0
    print(f"\nTotal: {total_lines} lines, {total_bytes_read/1e6:.1f}MB text, {len(all_tokens)} tokens in {elapsed:.1f}s")

    n_full = len(all_tokens) // seq_len
    total_tokens = n_full * seq_len
    all_tokens = all_tokens[:total_tokens]
    n_seqs = n_full
    print(f"Sequences: {n_seqs} x {seq_len} = {total_tokens} tokens")

    total_tokens = write_tokens(output_path, all_tokens, seq_len)

    file_size = output_path.stat().st_size
    print(f"\nWritten: {output_path}")
    print(f"File size: {file_size/1e6:.1f}MB ({file_size} bytes)")
    print(f"Total tokens: {total_tokens}")
    print(f"Throughput: {total_tokens/elapsed:.0f} tokens/s")

    first_ids = all_tokens[:min(100, len(all_tokens))]
    decoded = tokenizer.decode(first_ids)
    print(f"\nFirst 100 tokens decoded: {decoded[:200]}")

    meta_path = output_path.with_suffix(".meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"files_processed={files_processed}\n")
        f.write(f"total_lines={total_lines}\n")
        f.write(f"total_bytes_text={total_bytes_read}\n")
        f.write(f"total_tokens={total_tokens}\n")
        f.write(f"n_seqs={n_seqs}\n")
        f.write(f"seq_len={seq_len}\n")
        f.write(f"throughput_tokens_per_s={total_tokens/elapsed:.0f}\n")
    print(f"Metadata: {meta_path}")

if __name__ == "__main__":
    main()
