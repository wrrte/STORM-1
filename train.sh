export CUDA_VISIBLE_DEVICES=5

games=(
    Frostbite
    ChopperCommand
    Hero
    Asterix
    Gopher
    Jamesbond
    Kangaroo
    Krull
    KungFuMaster
    MsPacman
    Pong
    PrivateEye
    Qbert
    RoadRunner
    Seaquest
    UpNDown
    Alien
    Amidar
    Assault
    BankHeist
    BattleZone
    Boxing
    Breakout
    CrazyClimber
    DemonAttack
    Freeway
)

declare -A seed_map

for env_name in "${games[@]}"; do
    # Assign seed 1 for the first run, increment for subsequent runs
    if [[ -z "${seed_map[$env_name]}" ]]; then
        seed_map[$env_name]=1
    else
        seed_map[$env_name]=$((seed_map[$env_name] + 1))
    fi
    current_seed=${seed_map[$env_name]}

    echo "Starting training for ${env_name} with seed ${current_seed}..."
    python -u train.py \
        -n "${env_name}-life_done-wm_2L512D8H-100k-seed${current_seed}" \
        -seed ${current_seed} \
        -config_path "config_files/STORM.yaml" \
        -env_name "ALE/${env_name}-v5" \
        -trajectory_path "D_TRAJ/${env_name}.pkl" 
done
