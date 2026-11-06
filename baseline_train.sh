#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash baseline_train.sh GPU [--once | --max-jobs N]" >&2
    exit 2
fi
baseline_gpu="$1"
shift
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
baseline_state="${STORM_BASELINE_STATE:-results/baseline_campaign/state.json}"
baseline_python="${STORM_BASELINE_PYTHON:-}"
if [ -z "$baseline_python" ]; then
    if command -v python3 >/dev/null 2>&1; then
        baseline_bootstrap=python3
    elif command -v python >/dev/null 2>&1; then
        baseline_bootstrap=python
    else
        echo "Python is required. Set STORM_BASELINE_PYTHON to the STORM Python executable." >&2
        exit 127
    fi
    baseline_python="$("$baseline_bootstrap" -c \
        'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["python"])' \
        "$baseline_state")"
fi
exec "$baseline_python" -u baseline_campaign.py --state "$baseline_state" worker --gpu "$baseline_gpu" "$@"
