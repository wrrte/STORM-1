"""
Probing 실험용 데이터 수집 스크립트.

학습된 World Model + Agent를 로드하여 Seaquest 환경에서 rollout하면서,
매 스텝마다 (obs, RAM) 쌍을 수집합니다.

Usage:
    conda activate storm
    python collect_probing_data.py \
        --run_name Seaquest-life_done-wm_2L512D8H-100k-seed1_X \
        --num_episodes 50 \
        --output_dir probing_data
"""

import gymnasium
import ale_py
import argparse
import numpy as np
from einops import rearrange
import torch
from collections import deque
from tqdm import tqdm
import colorama
import os

from utils import seed_np_torch, load_config
import env_wrapper
import agents
from sub_models.world_models import WorldModel

# Gymnasium 1.0.0 이상에서 ALE 환경 등록
gymnasium.register_envs(ale_py)


def build_single_env(env_name, image_size):
    """단일 환경 생성 (RAM 접근 용이하도록 vec_env 사용하지 않음)."""
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    return env


def build_world_model(conf, action_dim):
    return WorldModel(
        in_channels=conf.Models.WorldModel.InChannels,
        action_dim=action_dim,
        transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
        transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
        transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
        transformer_num_heads=conf.Models.WorldModel.TransformerNumHeads
    ).cuda()


def build_agent(conf, action_dim):
    return agents.ActorCriticAgent(
        feat_dim=32*32 + conf.Models.WorldModel.TransformerHiddenDim,
        num_layers=conf.Models.Agent.NumLayers,
        hidden_dim=conf.Models.Agent.HiddenDim,
        action_dim=action_dim,
        gamma=conf.Models.Agent.Gamma,
        lambd=conf.Models.Agent.Lambda,
        entropy_coef=conf.Models.Agent.EntropyCoef,
    ).cuda()


def collect_data(env_name, image_size, num_episodes, world_model, agent, greedy=False):
    """
    학습된 agent를 사용하여 환경에서 rollout하고, (obs, RAM) 쌍을 수집합니다.
    eval.py의 eval_episodes와 동일한 추론 로직을 사용합니다.
    """
    world_model.eval()
    agent.eval()

    env = build_single_env(env_name, image_size)
    # ALE 환경에 접근하여 RAM을 읽기 위한 참조
    ale = env.unwrapped.ale

    all_obs = []
    all_rams = []
    episode_rewards = []

    for ep in tqdm(range(num_episodes), desc="Collecting episodes"):
        obs, info = env.reset()
        context_obs = deque(maxlen=16)
        context_action = deque(maxlen=16)
        ep_reward = 0
        done = False

        while not done:
            # --- 현재 상태 저장 ---
            ram_buffer = np.zeros(128, dtype=np.uint8)
            ale.getRAM(ram_buffer)
            current_ram = ram_buffer
            all_obs.append(obs.copy())  # (H, W, C) uint8
            all_rams.append(current_ram)

            # --- 행동 선택 (eval.py와 동일한 로직) ---
            # 단일 환경이므로 action을 np.array([action])으로 감싸서
            # vec_env (num_envs=1)과 동일한 shape (1,)로 맞춤
            with torch.no_grad():
                if len(context_action) == 0:
                    action = np.array([env.action_space.sample()])
                else:
                    obs_tensor = torch.cat(list(context_obs), dim=1)
                    context_latent = world_model.encode_obs(obs_tensor)
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).cuda()
                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(
                        context_latent, model_context_action
                    )
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=greedy
                    )

            # context 업데이트 (eval.py와 동일)
            obs_for_context = rearrange(
                torch.Tensor(obs).unsqueeze(0).cuda(), "B H W C -> B 1 C H W"
            ) / 255.0
            context_obs.append(obs_for_context)
            context_action.append(action)

            # 환경 진행: 단일 환경이므로 스칼라 action을 전달
            env_action = action.item() if isinstance(action, np.ndarray) else action
            obs, reward, terminated, truncated, info = env.step(env_action)
            ep_reward += reward
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        tqdm.write(f"  Episode {ep+1}: reward = {ep_reward:.0f}")

    env.close()

    all_obs = np.array(all_obs, dtype=np.uint8)
    all_rams = np.array(all_rams, dtype=np.uint8)

    print(f"\n{colorama.Fore.GREEN}수집 완료!{colorama.Style.RESET_ALL}")
    print(f"  총 프레임 수: {len(all_obs)}")
    print(f"  에피소드 수: {num_episodes}")
    print(f"  평균 보상: {np.mean(episode_rewards):.1f}")
    print(f"  obs shape: {all_obs.shape}")
    print(f"  ram shape: {all_rams.shape}")

    # 잠수부 개수 분포 출력 (RAM index 62)
    diver_counts = all_rams[:, 62]
    print(f"\n  잠수부 개수 분포 (RAM[62]):")
    for count in sorted(np.unique(diver_counts)):
        n = np.sum(diver_counts == count)
        print(f"    count={count}: {n} frames ({100*n/len(diver_counts):.1f}%)")

    return all_obs, all_rams, episode_rewards


def main():
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser(description="Probing 실험용 데이터 수집")
    parser.add_argument("--config_path", type=str, default="config_files/STORM.yaml",
                        help="STORM config 파일 경로")
    parser.add_argument("--env_name", type=str, default="ALE/Seaquest-v5",
                        help="환경 이름")
    parser.add_argument("--run_name", type=str, required=True,
                        help="체크포인트 디렉토리 이름 (예: Seaquest-life_done-wm_2L512D8H-100k-seed1_X)")
    parser.add_argument("--step", type=int, default=100000,
                        help="로드할 체크포인트 스텝")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="수집할 에피소드 수")
    parser.add_argument("--output_dir", type=str, default="probing_data",
                        help="출력 디렉토리")
    parser.add_argument("--greedy", action="store_true",
                        help="Greedy action 사용 (기본: stochastic)")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드")
    args = parser.parse_args()

    seed_np_torch(seed=args.seed)
    conf = load_config(args.config_path)

    # 환경으로부터 action_dim 확인
    dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize)
    action_dim = dummy_env.action_space.n
    print(f"Environment: {args.env_name}, action_dim: {action_dim}")
    dummy_env.close()

    # 모델 로드
    world_model = build_world_model(conf, action_dim)
    agent = build_agent(conf, action_dim)

    ckpt_dir = f"ckpt/{args.run_name}"
    wm_path = f"{ckpt_dir}/world_model_{args.step}.pth"
    agent_path = f"{ckpt_dir}/agent_{args.step}.pth"
    print(f"Loading world model: {wm_path}")
    print(f"Loading agent: {agent_path}")
    world_model.load_state_dict(torch.load(wm_path, map_location="cuda"))
    agent.load_state_dict(torch.load(agent_path, map_location="cuda"))

    # 데이터 수집
    all_obs, all_rams, episode_rewards = collect_data(
        env_name=args.env_name,
        image_size=conf.BasicSettings.ImageSize,
        num_episodes=args.num_episodes,
        world_model=world_model,
        agent=agent,
        greedy=args.greedy,
    )

    # 저장
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.run_name}_probing.npz")
    np.savez_compressed(
        output_path,
        obs=all_obs,
        ram=all_rams,
        episode_rewards=np.array(episode_rewards),
    )
    print(f"\n저장 완료: {output_path}")
    print(f"파일 크기: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
