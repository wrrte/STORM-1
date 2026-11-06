#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec bash queue_train.sh "$CUDA_VISIBLE_DEVICES" "$@"
