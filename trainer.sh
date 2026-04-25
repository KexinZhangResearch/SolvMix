#!/bin/bash
# SolvMix training launcher
# Usage:
#   ./trainer.sh
#   ./trainer.sh --config base
#   SOLVMIX_GPU=0 ./trainer.sh           # use GPU 0 only
#   SOLVMIX_GPU=0,1 ./trainer.sh         # use GPUs 0 and 1
#   SOLVMIX_GPU=2 ./trainer.sh --config base --custom_arg
#
# To switch dataset, either edit configs/base.yaml or create a new config.

cd "$(dirname "$0")/.."

# Optional: restrict visible GPUs via environment variable
if [ -n "$SOLVMIX_GPU" ]; then
    export CUDA_VISIBLE_DEVICES=$SOLVMIX_GPU
    echo "[trainer.sh] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

python -m SolvMix.trainer --config base "$@"
