#!/bin/bash

QUEUE_FILE="job_queue.txt"
START_SEED=1

# Original games list from the training scripts
games=(
    Gopher MsPacman Frostbite Hero ChopperCommand Jamesbond Asterix Qbert RoadRunner Seaquest UpNDown Alien Assault BankHeist BattleZone Boxing Breakout CrazyClimber DemonAttack Freeway Kangaroo Krull KungFuMaster Pong PrivateEye Amidar
)

echo "Populating $QUEUE_FILE with default games..."

for env_name in "${games[@]}"; do
    current_seed=${START_SEED}
    JOB="python -u train.py -n \"${env_name}-life_done-wm_2L512D8H-100k-seed${current_seed}\" -seed ${current_seed} -config_path \"config_files/STORM.yaml\" -env_name \"ALE/${env_name}-v5\" -trajectory_path \"D_TRAJ/${env_name}.pkl\" JointTrainAgent.Retrieval.enable False
python -u train.py -n \"${env_name}-life_done-wm_2L512D8H-100k-seed${current_seed}\" -seed ${current_seed} -config_path \"config_files/STORM.yaml\" -env_name \"ALE/${env_name}-v5\" -trajectory_path \"D_TRAJ/${env_name}.pkl\" JointTrainAgent.Retrieval.enable True JointTrainAgent.Retrieval.warmup_steps 20000
    "
    
    echo "$JOB" >> "$QUEUE_FILE"
    echo "Added: $env_name"
done

echo "Done! The queue now has $(wc -l < $QUEUE_FILE) jobs."
