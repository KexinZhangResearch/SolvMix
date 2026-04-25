#!/bin/bash
# SolvMix training launcher
# Usage:
#   ./SolvMix/trainer.sh
#   ./SolvMix/trainer.sh --config base
#
# To switch dataset, either edit SolvMix/configs/base.yaml or create a new config.

cd "$(dirname "$0")/.."
python -m SolvMix.trainer --config base "$@"
