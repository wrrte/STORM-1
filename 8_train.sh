# N_train.sh의 목적은 각 GPU에서 병렬로 학습을 수행하면서, 내가 원할 때 특정 GPU에서 학습을 중단할 수 있도록 하는 것이다. 따라서 실제로는 tmux 세션 8개에서 각각 하나의 N_train.sh를 실행한다. 내가 사용 가능한 GPU가 8개 모두가 아니라면 해당하는 N_train.sh만 실행하고, 실행 중간에 몇몇 GPU를 비우고 싶다면 해당 N_train.sh만 종료하는 식이다. 
# N_train.sh는 그저 업무가 중단되면 바로 실행 목록 중 가장 우선순위가 높은 걸 하나하나 실행하며, 따라서 이전에 다른 N_train.sh에서 실행하다가 중단된 작업이 있다면 자연스럽게 수행하게 된다.
# 이 파일은 주석을 다는 용이며, 일반적으로는 0~7_train.sh를 실행한다.

#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec bash queue_train.sh "$CUDA_VISIBLE_DEVICES" "$@"
