#!/bin/bash
# Launch a 4-way data-parallel pretraining run.
#   bash run_pretrain.sh <rank> <master_addr> [extra args]
# Override DIR to point at the directory holding the .bin shards and logs.
set -e
cd "$(dirname "$0")"

: "${DIR:=$(pwd)}"
: "${CONFIG:=dense200m}"
: "${STEPS:=300000}"
: "${BATCH_SIZE:=8}"
: "${SEQ_LEN:=1024}"
: "${WORLD:=4}"

python3 cyano_train.py \
  --rank "${1:?rank required}" --world "$WORLD" --master "${2:?master address required}" \
  --steps "$STEPS" --batch_size "$BATCH_SIZE" --seq_len "$SEQ_LEN" \
  --optimizer muon --z_loss 0 --config "$CONFIG" \
  --data "$DIR/train.bin" --eval_data "$DIR/eval.bin" \
  --logdir "$DIR/logs" --ckpt "$DIR/ckpt.pt" --tag "$CONFIG" \
  --save_every 1000 --eval_every 1000 --k 64 --sync_bucket_mb 64 \
  ${3:-}