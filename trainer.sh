#!/bin/bash
# SolvMix training launcher (Hydra)
# Usage:
#   ./trainer.sh                           # default config (base)
#   ./trainer.sh experiments=debug         # load experiment config
#   ./trainer.sh ++train.learning_rate=1e-3   # override learning rate
#   SOLVMIX_GPU=0 ./trainer.sh             # use GPU 0 only
#   SOLVMIX_GPU=0,1 ./trainer.sh           # use GPUs 0 and 1
#
# To switch dataset, edit configs/base.yaml or create a new experiment config.

cd "$(dirname "$0")/.."

# Optional: restrict visible GPUs via environment variable
if [ -n "$SOLVMIX_GPU" ]; then
    export CUDA_VISIBLE_DEVICES=$SOLVMIX_GPU
    echo "[trainer.sh] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

python -m SolvMix.trainer "$@"
