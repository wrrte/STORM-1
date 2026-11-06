#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash queue_train.sh GPU [QUEUE_FILE] [--once | --max-jobs N]" >&2
}

if (( $# < 1 )); then
    usage
    exit 2
fi
queue_gpu="$1"
shift
if [[ -z "$queue_gpu" || "$queue_gpu" == *,* ]]; then
    echo "Supply one GPU ID per worker." >&2
    exit 2
fi
queue_path=""
queue_once=false
queue_max_jobs=0
while (( $# )); do
    case "$1" in
        --once) queue_once=true; shift ;;
        --max-jobs)
            if (( $# < 2 )) || [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                usage
                exit 2
            fi
            queue_max_jobs="$2"
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        -*) usage; exit 2 ;;
        *)
            if [[ -n "$queue_path" ]]; then
                usage
                exit 2
            fi
            queue_path="$1"
            shift
            ;;
    esac
done
if $queue_once && (( queue_max_jobs )); then
    usage
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="$queue_gpu"
for dependency in nvidia-smi flock; do
    if ! command -v "$dependency" >/dev/null 2>&1; then
        echo "Required command not found: $dependency" >&2
        exit 1
    fi
done
queue_gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$queue_gpu")
if [[ -z "$queue_gpu_name" ]]; then
    echo "No GPU name returned for GPU $queue_gpu." >&2
    exit 1
fi
case "${queue_gpu_name,,}" in
    *a6000*) queue_suffix="A6000" ;;
    *3090*) queue_suffix="3090" ;;
    *"titan rtx"*) queue_suffix="titan" ;;
    *blackwell*6000*|*6000*blackwell*) queue_suffix="pro6k" ;;
    *) queue_suffix="default" ;;
esac
queue_path="${queue_path:-job_queue_${queue_suffix}.txt}"
queue_lock="${queue_path%.txt}.lock"
touch -- "$queue_path"
echo "Worker started on GPU $queue_gpu ($queue_gpu_name). Waiting for jobs in $queue_path..."

queue_completed=0
while true; do
    queue_job=""
    exec 200>"$queue_lock"
    flock -x 200
    sed -i '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "$queue_path"
    if [[ -s "$queue_path" ]]; then
        queue_job=$(head -n 1 -- "$queue_path")
        sed -i '1d' "$queue_path"
    fi
    flock -u 200
    exec 200>&-

    if [[ -n "$queue_job" ]]; then
        echo "[GPU $queue_gpu] Found job: $queue_job"
        # Keep each job's environment, working directory and exit local to it.
        queue_status=0
        ( eval "$queue_job" ) || queue_status=$?
        if (( queue_status )); then
            echo "[GPU $queue_gpu] Job failed (exit $queue_status). Continuing with the next job." >&2
        else
            echo "[GPU $queue_gpu] Job finished."
        fi
        queue_completed=$((queue_completed + 1))
        if $queue_once || (( queue_max_jobs && queue_completed >= queue_max_jobs )); then
            exit "$queue_status"
        fi
    elif $queue_once; then
        exit 0
    else
        sleep 10
    fi
done
