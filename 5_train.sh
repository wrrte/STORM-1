#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=5
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec bash baseline_train.sh "$CUDA_VISIBLE_DEVICES" "$@"
