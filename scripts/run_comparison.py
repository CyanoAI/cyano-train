#!/usr/bin/env python3
"""Run a controlled training comparison on homogeneous DCU pods.

This script is a harness, not a benchmark result generator. It records the
exact configuration, data hash, and timing window for each run so results
can be compared without mixing accounting methods.

Usage:
    python run_comparison.py --frameworks cyano_train,nanotron \
        --config dense05b --seq-len 1024 --batch 2 --steps 200
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

KUBECONFIG = "/work/home/scnyvpjab9/.kube/20078-k8s1/account-2069689184693813250-564d5c5b/config"
LIB = "/opt/dtk-26.04/hip/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib:/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/dtk-26.04/.hyhal/lib"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kubectl(args, timeout=60):
    env = os.environ.copy()
    env["KUBECONFIG"] = KUBECONFIG
    return subprocess.run(["kubectl", *args], env=env, text=True,
                          capture_output=True, timeout=timeout)


def list_pods(gpu_model="Z200SM_80", minimum=2):
    result = kubectl(["get", "pods", "-o", "json"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    pods = []
    for item in json.loads(result.stdout)["items"]:
        name = item["metadata"]["name"]
        if item["status"].get("phase") != "Running":
            continue
        probe = kubectl(["exec", name, "--", "env", f"LD_LIBRARY_PATH={LIB}",
                         "python3", "-c",
                         "import torch; print(torch.cuda.get_device_name(0))"], timeout=30)
        if probe.returncode == 0 and probe.stdout.strip() == gpu_model:
            pods.append(name)
    if len(pods) < minimum:
        raise RuntimeError(f"need {minimum} {gpu_model} pods, found {len(pods)}: {pods}")
    return pods


def build_command(framework, args, rank, world, master):
    base = ["--rank", str(rank), "--world", str(world), "--master", master,
            "--steps", str(args.steps), "--max_run_steps", str(args.steps),
            "--batch_size", str(args.batch), "--seq_len", str(args.seq_len),
            "--warmup", str(args.warmup), "--eval_every", str(args.eval_every),
            "--save_every", "0", "--data", args.data, "--logdir", args.logdir,
            "--ckpt", args.ckpt, "--tag", args.tag,
            "--ckpt_save_optimizer", "0"]
    if framework == "cyano_train":
        return ["python3", args.train_script, "--optimizer", args.optimizer,
                "--config", args.config, *base]
    raise NotImplementedError(f"framework {framework} is not wired up yet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frameworks", default="cyano_train")
    parser.add_argument("--train-script", default="/tmp/cyano_train.py")
    parser.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    parser.add_argument("--config", default="dense05b")
    parser.add_argument("--data", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--logdir", default="/tmp/comparison_logs")
    parser.add_argument("--ckpt", default="/tmp/comparison.pt")
    parser.add_argument("--tag", default="comparison")
    parser.add_argument("--gpu-model", default="Z200SM_80")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pods = list_pods(args.gpu_model, 2)
    master = kubectl(["exec", pods[0], "--", "hostname", "-i"],
                     timeout=30).stdout.strip().split()[0]
    record = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "frameworks": args.frameworks.split(","),
        "pods": pods,
        "gpu_model": args.gpu_model,
        "master": master,
        "config": {
            "model": args.config, "seq_len": args.seq_len, "batch": args.batch,
            "steps": args.steps, "optimizer": args.optimizer,
            "warmup": args.warmup, "eval_every": args.eval_every,
        },
        "data": {"path": args.data, "sha256": sha256(args.data)},
        "accounting": "aggregate tokens from successful optimizer updates / wall time",
    }
    print(json.dumps(record, indent=2))
    if args.dry_run:
        return

    env = os.environ.copy()
    env["KUBECONFIG"] = KUBECONFIG
    procs = []
    for rank, pod in enumerate(pods):
        cmd = ["kubectl", "exec", pod, "--", "env", "LD_LIBRARY_PATH=" + LIB,
               *build_command(args.frameworks.split(",")[0], args, rank,
                              len(pods), master)]
        procs.append((rank, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True)))
    failed = False
    for rank, proc in procs:
        output, _ = proc.communicate(timeout=3600)
        print(f"=== rank {rank} exit {proc.returncode} ===")
        print(output[-4000:])
        failed |= proc.returncode != 0
    record["completed_at"] = datetime.now(timezone.utc).isoformat()
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    Path(args.logdir, "run_record.json").write_text(json.dumps(record, indent=2))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
