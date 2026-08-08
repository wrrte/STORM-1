#!/bin/bash
# 5번 GPU가 존재하는지 확인 (에러 없이 실행되면 존재하는 것)
if nvidia-smi -i 5 >/dev/null 2>&1; then
    export CUDA_VISIBLE_DEVICES=5
    echo "✅ GPU 5 is detected. Using GPU 5 (Robone)."
else
    export CUDA_VISIBLE_DEVICES=3
    echo "⚠️ GPU 5 not found. Falling back to GPU 3 (B200)."
fi

env_name=Frostbite
python -u train.py \
    -n "${env_name}-life_done-wm_2L512D8H-100k-seed1" \
    -seed 1 \
    -config_path "config_files/STORM.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" 

env_name=Hero
python -u train.py \
    -n "${env_name}-life_done-wm_2L512D8H-100k-seed1" \
    -seed 1 \
    -config_path "config_files/STORM.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" 

env_name=ChopperCommand
python -u train.py \
    -n "${env_name}-life_done-wm_2L512D8H-100k-seed1" \
    -seed 1 \
    -config_path "config_files/STORM.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" 

env_name=BankHeist
python -u train.py \
    -n "${env_name}-life_done-wm_2L512D8H-100k-seed1" \
    -seed 1 \
    -config_path "config_files/STORM.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" 

env_name=PrivateEye
python -u train.py \
    -n "${env_name}-life_done-wm_2L512D8H-100k-seed1" \
    -seed 1 \
    -config_path "config_files/STORM.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" 