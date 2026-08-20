#!/bin/bash

QUEUE_FILE="job_queue.txt"
START_SEED=2000

# Original games list from the training scripts
games=(
    BattleZone Alien Assault BankHeist Kangaroo Krull KungFuMaster Pong
)

echo "Populating $QUEUE_FILE with default games..."

for env_name in "${games[@]}"; do
    current_seed1=${START_SEED}
    current_seed1=$((START_SEED + 10))
    JOB="python -u train.py -n \"${env_name}-life_done-wm_2L512D8H-100k-seed${current_seed1}\" -seed ${current_seed1} -config_path \"config_files/STORM.yaml\" -env_name \"ALE/${env_name}-v5\" -trajectory_path \"D_TRAJ/${env_name}.pkl\" JointTrainAgent.Retrieval.enable True
    "
    
    echo "$JOB" >> "$QUEUE_FILE"
    echo "Added: $env_name"
done

echo "Done! The queue now has $(wc -l < $QUEUE_FILE) jobs."
