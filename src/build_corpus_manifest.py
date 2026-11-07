#!/usr/bin/env python3
"""Build a deduplicated, approved-only corpus manifest.

Design (validated against 295 committed_chunks.jsonl / 435,743 records):
  * dedup key = bin sha256 (content identity). source inode+size alone is WRONG:
    it over-counts by ~19.1B because chunk re-slicing makes `index` unstable.
  * keep only the newest generation when several generations carry the same sha256.
  * require training_approved is True (the field must exist; we synthesize it only
    from an explicit allow-list file, never implicitly).

Usage:
  python3 build_corpus_manifest.py \
    --root /path/to/tok_full \
    --approval approvals.json \
    --out corpus_manifest.jsonl --summary corpus_summary.json

approvals.json format:
  {"policy_sha256": ["<approved policy sha>", ...],
   "domains": {"v5c_code_github_code_clean_s0": "code", ...}}
Records whose policy_sha256 is not listed are marked training_approved=false and
excluded from the approved stream. Nothing is deleted; this only writes a manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path

GEN_ORDER = ["v3", "v4", "v4b", "v5b", "v5c"]


def verified_bin(root, domain_dir, relative_name, expected_bytes, expected_sha):
    domain_root = (root / domain_dir).resolve()
    path = (domain_root / relative_name).resolve()
    try:
        path.relative_to(domain_root)
    except ValueError:
        return None
    if not path.is_file() or path.stat().st_size != expected_bytes:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha:
        return None
    return path


def generation_of(dirname):
    for g in reversed(GEN_ORDER):
        if dirname.startswith(g + "_"):
            return g
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--approval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    a = ap.parse_args()

    root = Path(a.root)
    approval = json.loads(Path(a.approval).read_text())
    approved_policies = set(approval.get("policy_sha256", []))
    domain_override = approval.get("domains", {})

    best = {}  # sha256 -> record
    scanned = 0
    for manifest in sorted(root.glob("*/committed_chunks.jsonl")):
        domain_dir = manifest.parent.name
        gen = generation_of(domain_dir)
        for line in manifest.read_text().splitlines():
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            scanned += 1
            sha = obj.get("sha256")
            if not sha:
                continue
            if verified_bin(root, domain_dir, obj.get("bin", ""),
                           int(obj.get("bytes") or 0), sha) is None:
                continue
            prev = best.get(sha)
            if prev is not None:
                if GEN_ORDER.index(gen) <= GEN_ORDER.index(prev["_gen"]):
                    continue
            obj["_gen"] = gen
            obj["_domain_dir"] = domain_dir
            obj["_domain"] = domain_override.get(domain_dir, domain_dir)
            obj["training_approved"] = obj.get("policy_sha256") in approved_policies
            best[sha] = obj

    approved = [r for r in best.values() if r["training_approved"]]
    total_tokens = sum(int(r.get("tokens") or 0) for r in approved)
    by_domain = {}
    by_gen = {}
    for r in approved:
        d = r["_domain"]
        by_domain[d] = by_domain.get(d, 0) + int(r.get("tokens") or 0)
        by_gen[r["_gen"]] = by_gen.get(r["_gen"], 0) + int(r.get("tokens") or 0)

    with open(a.out, "w") as f:
        for r in sorted(approved, key=lambda x: -int(x.get("tokens") or 0)):
            rec = {
                "bin": str(Path(r["_domain_dir"]) / r["bin"]),
                "bin_sha256": r["sha256"],
                "tokens": int(r.get("tokens") or 0),
                "bytes": int(r.get("bytes") or 0),
                "dedup_key": "bin_sha256",
                "dedup_group": r["sha256"],
                "source_inode_size": "%s:%s:%s" % (
                    r.get("source", {}).get("device"),
                    r.get("source", {}).get("inode"),
                    r.get("source", {}).get("size")),
                "generation": r["_gen"],
                "domain": r["_domain"],
                "policy_sha256": r.get("policy_sha256"),
                "tokenizer_sha256": r.get("tokenizer_sha256"),
                "training_approved": True,
                "license": r.get("license"),
                "contamination_index": r.get("contamination_index"),
            }
            f.write(json.dumps(rec) + "\n")

    summary = {
        "scanned_records": scanned,
        "unique_bin_sha256": len(best),
        "approved_records": len(approved),
        "approved_tokens": total_tokens,
        "approved_by_domain": dict(sorted(by_domain.items(), key=lambda kv: -kv[1])),
        "approved_by_generation": by_gen,
        "approved_policies": sorted(approved_policies),
        "note": "records not matching approved_policies are excluded, not deleted",
    }
    Path(a.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
