#!/usr/bin/env bash
# Run inside storm. The lower bound is fixed at 3.5; other defaults can be overridden.
set -euo pipefail

td_search_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
td_project_root="$(cd -- "${td_search_dir}/../.." && pwd)"
cd -- "${td_project_root}"

exec python "${td_search_dir}/find_td_error_threshold.py" \
  --checkpoint STORM-1/ckpt/Boxing-6040_Shared \
  --gpus 0 1 2 3 4 5 \
  --upper 4.0 --coarse-points 6 --refinement-rounds 0 \
  --target 5.99 --tolerance 0.05 \
  --steps 500 --validation-steps 0 \
  --output STORM-1/results/td_error_only_threshold_search/boxing_gpus0-5_3.5-4.0_step0.1_500updates \
  "$@" --lower 3.5
