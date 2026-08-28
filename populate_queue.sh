#!/bin/bash

QUEUE_FILE="job_queue.txt"
START_SEED=6000

# Original games list from the training scripts
# Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Frostbite Gopher Hero JamesBond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown
games=(
    Frostbite UpNDown Qbert BattleZone Alien Amidar Assault Asterix BankHeist  Boxing Breakout ChopperCommand CrazyClimber DemonAttack Gopher Hero JamesBond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye RoadRunner Seaquest 
)

echo "Populating $QUEUE_FILE with default games..."

for env_name in "${games[@]}"; do
    current_seed1=${START_SEED}
    current_seed2=$((START_SEED + 10))
    current_seed3=$((START_SEED + 20))
    JOB="python -u train.py -n \"${env_name}-life_done-wm_2L512D8H-100k-seed${current_seed1}\" -seed ${current_seed1} -config_path \"config_files/STORM.yaml\" -env_name \"ALE/${env_name}-v5\" -trajectory_path \"D_TRAJ/${env_name}.pkl\" JointTrainAgent.Retrieval.enable False
python -u train.py -n \"${env_name}-life_done-wm_2L512D8H-100k-seed${current_seed1}\" -seed ${current_seed1} -config_path \"config_files/STORM.yaml\" -env_name \"ALE/${env_name}-v5\" -trajectory_path \"D_TRAJ/${env_name}.pkl\" JointTrainAgent.Retrieval.enable True

python -u train.py -n \"${env_name}-life_done-wm_2L512D8H-100k-seed${current_seed2}\" -seed ${current_seed2} -config_path \"config_files/STORM.yaml\" -env_name \"ALE/${env_name}-v5\" -trajectory_path \"D_TRAJ/${env_name}.pkl\" JointTrainAgent.Retrieval.enable False
python -u train.py -n \"${env_name}-life_done-wm_2L512D8H-100k-seed${current_seed2}\" -seed ${current_seed2} -config_path \"config_files/STORM.yaml\" -env_name \"ALE/${env_name}-v5\" -trajectory_path \"D_TRAJ/${env_name}.pkl\" JointTrainAgent.Retrieval.enable True

python -u train.py -n \"${env_name}-life_done-wm_2L512D8H-100k-seed${current_seed3}\" -seed ${current_seed3} -config_path \"config_files/STORM.yaml\" -env_name \"ALE/${env_name}-v5\" -trajectory_path \"D_TRAJ/${env_name}.pkl\" JointTrainAgent.Retrieval.enable False
python -u train.py -n \"${env_name}-life_done-wm_2L512D8H-100k-seed${current_seed3}\" -seed ${current_seed3} -config_path \"config_files/STORM.yaml\" -env_name \"ALE/${env_name}-v5\" -trajectory_path \"D_TRAJ/${env_name}.pkl\" JointTrainAgent.Retrieval.enable True

"
    
    echo "$JOB" >> "$QUEUE_FILE"
    echo "Added: $env_name"
done

echo "Done! The queue now has $(wc -l < $QUEUE_FILE) jobs."
