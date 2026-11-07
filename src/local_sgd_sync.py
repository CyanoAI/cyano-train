import torch
import torch.distributed as dist


@torch.no_grad()
def average_masters(entries, world, bucket_mb):
    capacity = max(1, int(bucket_mb * 1024 * 1024 / 4))
    pending = []
    size = 0

    def flush():
        if not pending:
            return
        flat = torch.cat([entry["master"].reshape(-1) for entry in pending])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world)
        offset = 0
        for entry in pending:
            count = entry["master"].numel()
            entry["master"].copy_(flat[offset:offset + count].view_as(entry["master"]))
            entry["output"].copy_(entry["master"])
            entry["parameter"].copy_(entry["output"])
            offset += count

    for entry in entries:
        count = entry["master"].numel()
        if pending and size + count > capacity:
            flush()
            pending = []
            size = 0
        pending.append(entry)
        size += count
    flush()
