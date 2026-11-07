#!/usr/bin/env python3
"""Clean implementation of v3 candidate mixer (document level shuffle).

Usage:
  python3 train_mix_v3.py \
    --root /path/to/tok_full \
    --out ./mix.bin --recipe zh+ --target-tokens 1300000000

Outputs: <out>.bin and <out>.manifest.json.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

EOS = 50256

RECIPES = {
    "zh+": {"code": .38, "book": .24, "math": .14, "wiki": .10,
             "lit": .07, "zh": .07},
    "uniform": {},
    "code50": {"code": .50, "book": .20, "math": .14, "wiki": .08,
                "lit": .06, "zh": .02},
}
BUCKET_OF = {"paper": "book", "other": "book"}  # stories+gutenberg ~ book-like


def committed_bins(root):
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("v3_")):
        dom = d.name[3:]
        mf = d / "committed_chunks.jsonl"
        if not mf.exists():
            continue
        bins = []
        for line in mf.read_text().splitlines():
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("training_approved") is not True:
                continue
            if not obj.get("sha256"):
                continue
            domain_root = d.resolve()
            pb = (domain_root / obj["bin"]).resolve()
            try:
                pb.relative_to(domain_root)
            except ValueError:
                continue
            digest = hashlib.sha256()
            if pb.is_file():
                with pb.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
            if (pb.is_file() and pb.stat().st_size == obj["tokens"] * 4
                    and digest.hexdigest() == obj["sha256"]):
                bins.append(pb)
        if bins:
            out[dom] = bins
    return out


def split_docs(arr):
    cuts = np.flatnonzero(arr == EOS) + 1
    if not cuts.size or cuts[-1] < arr.size:
        cuts = np.append(cuts, arr.size)
    bounds = [0] + cuts.tolist()
    return [arr[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", default="zh+", choices=list(RECIPES))
    ap.add_argument("--target-tokens", type=int, default=1_300_000_000)
    a = ap.parse_args()
    recipe = RECIPES[a.recipe]
    root = Path(a.root)
    domain_bins = committed_bins(root)
    if not domain_bins:
        ap.error("no v3_* domains with candidate chunks found")

    if a.recipe == "uniform":
        fracs = {d: 1.0 / len(domain_bins) for d in domain_bins}
    else:
        fracs = {d: recipe.get(BUCKET_OF.get(d, d), 0.0) for d in domain_bins}
        s = sum(fracs.values())
        if s == 0:
            ap.error("recipe has no overlap with available domains")
        fracs = {k: v / s for k, v in fracs.items()}

    rng = random.Random(42)
    out = Path(a.out)
    manifest = {"recipe": a.recipe, "fracs": fracs, "domains": {}, "target_tokens": a.target_tokens}
    selected = []
    domain_stats = {}
    wrote_total = 0
    with open(str(out) + ".tmp", "wb") as fo:
        for dom, bins in sorted(domain_bins.items()):
            quota = int(a.target_tokens * fracs[dom])
            arrs = [np.fromfile(p, dtype=np.uint32) for p in bins]
            docs = []
            for arr in arrs:
                docs.extend(split_docs(arr))
            order = list(range(len(docs)))
            rng.shuffle(order)
            take = 0
            for i in order:
                if take >= quota:
                    break
                doc = docs[i]
                selected.append(doc)
                take += int(doc.size)
            domain_stats[dom] = {"wrote_tokens": take, "quota": quota,
                                "docs_total": len(docs), "bins": len(bins)}
            wrote_total += take
            print(f"dom={dom} wrote={take} quota={quota} docs={len(docs)}")
        rng.shuffle(selected)
        for doc in selected:
            doc.tofile(fo)
    manifest["domains"] = domain_stats
    os.replace(str(out) + ".tmp", str(out))
    manifest["total_tokens"] = wrote_total
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print("total", wrote_total, "->", out)


if __name__ == "__main__":
    main()
