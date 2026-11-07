"""Per-rank checkpoints for synchronized callers sharing a checkpoint namespace.

``path + '.manifest.json'`` is the commit point. Calls must not overlap saves
or loads on the same path. With a process group, rank/world are group-local.
State is a dict containing a nonnegative integer ``step``. Only load trusted
checkpoints: PyTorch pickle loading is explicitly enabled.
"""

import json
import os

import torch
import torch.distributed as dist


def _validate_identity(rank, world):
    if type(world) is not int or world < 1:
        raise ValueError("world must be a positive integer")
    if type(rank) is not int or not 0 <= rank < world:
        raise ValueError("rank must be an integer in [0, world)")


def _read_manifest(path, world):
    try:
        with open(path + ".manifest.json", "r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except FileNotFoundError:
        return None
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    for key in ("version", "world", "step", "slot"):
        if type(manifest.get(key)) is not int:
            raise ValueError(f"manifest {key} must be an integer")
    if manifest["version"] != 1:
        raise ValueError("unsupported manifest version")
    if manifest["world"] < 1 or manifest["world"] != world:
        raise ValueError("manifest world mismatch")
    if manifest["step"] < 0:
        raise ValueError("manifest step must be nonnegative")
    if manifest["slot"] not in (0, 1):
        raise ValueError("manifest slot must be 0 or 1")
    return manifest


def _gather(value, count, group):
    if count == 1:
        return [value]
    values = [None] * count
    dist.all_gather_object(values, value, group=group)
    return values


def save_rank_checkpoint(path, state, rank, world, group=None):
    """Collectively save each rank; return None after manifest publication.

    All group members must participate. Recoverable local I/O/validation errors
    become RuntimeError on every rank; a dead process or failed communicator
    still requires the caller's distributed timeout/restart handling.
    """
    initialized = dist.is_available() and dist.is_initialized()
    count = dist.get_world_size(group) if initialized else 1
    status = {"error": None}
    try:
        _validate_identity(rank, world)
        if world != count or (world > 1 and not initialized):
            raise ValueError("world does not match the initialized process group")
        actual_rank = dist.get_rank(group) if initialized else 0
        if rank != actual_rank:
            raise ValueError("rank does not match the process group")
        path = os.fsdecode(path)
        if not isinstance(state, dict):
            raise ValueError("state must be a dict")
        step = state.get("step")
        if type(step) is not int or step < 0:
            raise ValueError("state step must be a nonnegative integer")
        previous = _read_manifest(path, world)
        status.update(step=step, previous=previous)
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"

    statuses = _gather(status, count, group)
    errors = [f"rank {i}: {s['error']}" for i, s in enumerate(statuses) if s["error"]]
    if errors:
        raise RuntimeError("checkpoint preparation failed: " + "; ".join(errors))
    if any(s != statuses[0] for s in statuses[1:]):
        raise RuntimeError("checkpoint step or previous manifest differs across ranks")

    slot = 0 if previous is None else 1 - previous["slot"]
    error = None
    try:
        with open(f"{path}.slot{slot}.rank{rank}.pt", "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = _gather(error, count, group)
    if any(error is not None for error in errors):
        details = "; ".join(f"rank {i}: {error}" for i, error in enumerate(errors) if error is not None)
        raise RuntimeError("checkpoint storage failed: " + details)

    publication = [None]
    if rank == 0:
        try:
            manifest = {"version": 1, "world": world, "step": step, "slot": slot}
            target = path + ".manifest.json"
            with open(target + ".tmp", "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(target + ".tmp", target)
        except Exception as exc:
            publication[0] = f"{type(exc).__name__}: {exc}"
    if count > 1:
        source = dist.get_global_rank(group, 0) if group is not None else 0
        dist.broadcast_object_list(publication, src=source, group=group)
    if publication[0] is not None:
        raise RuntimeError("checkpoint manifest publication failed: " + publication[0])


def load_rank_checkpoint(path, rank, world, map_location="cpu"):
    """Load this rank's committed state, or mark a legacy load as inexact."""
    _validate_identity(rank, world)
    path = os.fsdecode(path)
    manifest = _read_manifest(path, world)
    if manifest is None:
        with open(path, "rb") as stream:
            state = torch.load(stream, map_location=map_location, weights_only=False)
        state["legacy_resume"] = True
        return state
    slot = manifest["slot"]
    with open(f"{path}.slot{slot}.rank{rank}.pt", "rb") as stream:
        state = torch.load(stream, map_location=map_location, weights_only=False)
    if not isinstance(state, dict) or type(state.get("step")) is not int or state["step"] != manifest["step"]:
        raise ValueError("rank checkpoint step does not match manifest")
    return state
