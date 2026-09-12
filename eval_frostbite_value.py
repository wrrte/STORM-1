import numpy as np
import cv2
import torch
import argparse
import json
import random
from pathlib import Path
from frostbite_value_stats import analyze_values, make_blocks

conf = None

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_igloo_stage(ram_77_val):
    if ram_77_val == 255:
        return 0
    else:
        return ram_77_val + 1

def load_models(run_name, step, env_name):
    import gymnasium
    from train import build_world_model, build_agent
    import ale_py
    gymnasium.register_envs(ale_py)
    dummy_env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    action_dim = dummy_env.action_space.n
    dummy_env.close()

    world_model = build_world_model(conf, action_dim)
    agent = build_agent(conf, action_dim)

    wm_path = f"ckpt/{run_name}/world_model_{step}.pth"
    agent_path = f"ckpt/{run_name}/agent_{step}.pth"
    
    world_model.load_state_dict(torch.load(wm_path, map_location="cuda"))
    agent.load_state_dict(torch.load(agent_path, map_location="cuda"))
    
    world_model.eval()
    agent.eval()
    
    for param in world_model.parameters():
        param.requires_grad = False
    for param in agent.parameters():
        param.requires_grad = False
        
    return world_model, agent

def collect_episodes_data(env_name, world_model, agent, target_frames=20000, seed=2027):
    import gymnasium
    import ale_py
    from collections import deque
    from env_wrapper import MaxLast2FrameSkipWrapper
    
    gymnasium.register_envs(ale_py)
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=conf.BasicSettings.ImageSize)
    
    all_obs = []
    all_action = []
    all_ram = []
    episode_ids, episode_steps, episode_ends = [], [], []
    episode_id, episode_step = 0, 0
    
    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed)
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)
    
    from tqdm import tqdm
    print(f"\n[{env_name}] 동적 데이터 수집 시작 (목표: {target_frames} 프레임)...")
    for step in tqdm(range(target_frames), desc="Collecting Rollouts"):
        obs_np = np.array(obs) # (64, 64, 3)
        ram = env.unwrapped.ale.getRAM()
        
        with torch.no_grad():
            if len(context_action) == 0:
                action = env.action_space.sample()
            else:
                context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=1) # (1, seq)
                model_context_action = torch.Tensor(model_context_action).cuda()
                
                prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                action_batch = agent.sample_as_env_action(
                    torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                    greedy=False
                )
                action = action_batch[0]
                
        all_obs.append(obs_np)
        all_action.append(action)
        all_ram.append(ram.copy())
        episode_ids.append(episode_id)
        episode_steps.append(episode_step)
        
        # obs_tensor needs to be (1, 1, 3, 64, 64) float32 in [0, 1]
        obs_tensor = torch.Tensor(obs_np).cuda().unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 2, 3) / 255.0
        context_obs.append(obs_tensor)
        context_action.append([action])
        
        obs, reward, done, truncated, info = env.step(action)
        
        episode_ends.append(bool(done or truncated))
        episode_step += 1
        if done or truncated:
            episode_id += 1
            episode_step = 0
            obs, info = env.reset()
            context_obs.clear()
            context_action.clear()
            
    print(f"데이터 수집 완료!")
    env.close()
    return dict(obs=np.array(all_obs), action=np.array(all_action).reshape(-1),
                ram=np.array(all_ram), episode_id=np.array(episode_ids),
                episode_step=np.array(episode_steps), episode_end=np.array(episode_ends))

def evaluate_values(world_model, agent, data, templates, valid_blocks, seed):
    obs, action = data["obs"], data["action"]
    # 4. 가치 함수 평가 (캐시 재사용 최적화)
    all_values = [] # 각 유효 프레임마다의 17개 가치를 저장
    
    from tqdm import tqdm
    print("\n가치 함수(Value Function) 평가 중...")
    for block_id, (start, end) in enumerate(tqdm(valid_blocks, desc="Evaluating blocks")):
        block_length = end - start
        num_eval_frames = block_length - 8
        
        base_obs = obs[start:end].copy()
        base_action = action[start:end-1].copy()
        
        base_action_t = torch.from_numpy(base_action).long().cuda().unsqueeze(0) # (1, block_length-1)
        
        # 특정 블록 내에서 17개의 평가를 진행
        # 구조: block_values[frame_idx][stg]
        block_values = [[] for _ in range(num_eval_frames)]
        
        for stg in range(17):
            # Common random numbers across stage variants and both models.
            seed_all(seed + block_id)
            mod_obs = base_obs.copy()
            mod_obs[:, 10:17, 42:58, :] = templates[stg]
            mod_obs_t = torch.from_numpy(mod_obs).permute(0, 3, 1, 2).unsqueeze(0).float().cuda() / 255.0
            
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    context_latent = world_model.encode_obs(mod_obs_t)
                    world_model.storm_transformer.reset_kv_cache_list(1, dtype=torch.bfloat16)
                    
                    last_dist_feat = None
                    
                    # 4.1. 앞의 8프레임(0~7)으로 트랜스포머 초기 컨텍스트 예열
                    for j in range(8):
                        _, _, _, _, dist_feat = world_model.predict_next(
                            context_latent[:, j:j+1],
                            base_action_t[:, j:j+1],
                            log_video=False
                        )
                        last_dist_feat = dist_feat
                        
                    # 4.2. 9번째 프레임부터 나머지 모든 프레임을 캐시를 유지한 채 순차 평가
                    for eval_idx in range(num_eval_frames):
                        frame_idx = 8 + eval_idx
                        
                        # 현재 프레임 가치 평가
                        state = torch.cat([context_latent[:, frame_idx:frame_idx+1], last_dist_feat], dim=-1)
                        value = agent.value(state)
                        block_values[eval_idx].append(value.item())
                        
                        # 다음 프레임이 남아있다면 1스텝 전진하여 캐시 누적
                        if frame_idx < block_length - 1:
                            _, _, _, _, dist_feat = world_model.predict_next(
                                context_latent[:, frame_idx:frame_idx+1],
                                base_action_t[:, frame_idx:frame_idx+1],
                                log_video=False
                            )
                            last_dist_feat = dist_feat
                            
        all_values.extend(block_values)
        
    all_values = np.array(all_values) # (총 평가 프레임 수, 17)
    print(f"총 {len(all_values)}개의 유효 프레임에 대해 평가 완료.")
    
    return all_values


def main():
    parser = argparse.ArgumentParser(description="Paired Frostbite critic analysis (no training).")
    parser.add_argument("--baseline-run", "--run_name", dest="baseline_run")
    parser.add_argument("--flash-run")
    parser.add_argument("--step", type=int, default=100000)
    parser.add_argument("--frames", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--config", default="config_files/STORM.yaml")
    parser.add_argument("--template-dir", default="frostbite_igloo_templates")
    parser.add_argument("--data", help="Shared NPZ; loaded if present, otherwise collected with baseline and saved.")
    parser.add_argument("--output", default="results/frostbite_value_comparison")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-unit", choices=["episode", "block"], default="episode")
    parser.add_argument("--analyze-only", action="store_true", help="Recompute statistics from saved paired_values.npz; no GPU evaluation.")
    args = parser.parse_args()
    if args.bootstrap < 100 or args.frames < 9:
        parser.error("--bootstrap must be >=100 and --frames >=9")
    if args.seed < 0 or args.seed + args.frames >= 2**32:
        parser.error("--seed must be nonnegative and --seed + --frames < 2**32")
    output = Path(args.output)
    if args.analyze_only:
        analyze_values(output, args.bootstrap, args.seed, args.bootstrap_unit)
        return
    if not args.baseline_run or not args.flash_run:
        parser.error("Both --baseline-run and --flash-run are required")
    if (output / "paired_values.npz").exists():
        parser.error("Results already exist: use --analyze-only or a new --output directory")
    templates = {}
    for stage in range(17):
        path = Path(args.template_dir) / f"stage_{stage:02d}.png"
        img = cv2.imread(str(path))
        if img is None or img.shape != (7, 16, 3):
            parser.error(f"Expected a 7x16 RGB template: {path}")
        templates[stage] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if not torch.cuda.is_available():
        parser.error("CUDA is required for model evaluation")
    global conf
    from utils import load_config
    conf = load_config(args.config)
    # Check both checkpoint pairs before any expensive collection.
    for run in (args.baseline_run, args.flash_run):
        for kind in ("world_model", "agent"):
            path = Path("ckpt") / run / f"{kind}_{args.step}.pth"
            if not path.is_file():
                parser.error(f"Missing checkpoint: {path}")
    output.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data) if args.data else output / "shared_contexts.npz"
    seed_all(args.seed)
    world_model, agent = load_models(args.baseline_run, args.step, "ALE/Frostbite-v5")
    if data_path.exists():
        with np.load(data_path, allow_pickle=False) as stored:
            data = {key: stored[key] for key in stored.files}
    else:
        data = collect_episodes_data("ALE/Frostbite-v5", world_model, agent, args.frames, args.seed)
        data["collection_run"] = np.array(args.baseline_run)
        data["collection_step"] = np.array(args.step)
        data["collection_seed"] = np.array(args.seed)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data_path, **data)
    required = ("obs", "action", "ram", "episode_id", "episode_step", "episode_end")
    if any(key not in data for key in required):
        raise ValueError("Shared data must include observations, actions, RAM, episode IDs, steps and ends")
    n = len(data["obs"])
    if any(len(data[key]) != n for key in required) or data["obs"].shape[1:] != (64, 64, 3):
        raise ValueError("Invalid shared data shapes")
    data["action"] = data["action"].reshape(-1)
    stages = np.array([get_igloo_stage(r[77]) for r in data["ram"]])
    blocks = make_blocks(stages, data["episode_id"])
    if not blocks:
        raise ValueError("No valid blocks of >=9 frames; collect more frames")
    baseline = evaluate_values(world_model, agent, data, templates, blocks, args.seed)
    print("Baseline 평가 완료. 같은 문맥으로 FLASH를 평가합니다.")
    del world_model, agent
    torch.cuda.empty_cache()
    seed_all(args.seed)
    world_model, agent = load_models(args.flash_run, args.step, "ALE/Frostbite-v5")
    flash = evaluate_values(world_model, agent, data, templates, blocks, args.seed)
    frame_ids = np.concatenate([np.arange(start + 8, end) for start, end in blocks])
    block_ids = np.concatenate([np.full(end - start - 8, i) for i, (start, end) in enumerate(blocks)])
    np.savez_compressed(output / "paired_values.npz", baseline=baseline, flash=flash,
                        frame_id=frame_ids, block_id=block_ids,
                        episode_id=data["episode_id"][frame_ids],
                        episode_step=data["episode_step"][frame_ids])
    np.savez_compressed(output / "used_templates.npz", templates=np.stack([templates[i] for i in range(17)]))
    protocol = dict(vars(args), data_path=str(data_path.resolve()),
                    collection_run=str(data.get("collection_run", "unknown")),
                    collection_step=int(data["collection_step"]) if "collection_step" in data else None,
                    collection_seed=int(data["collection_seed"]) if "collection_seed" in data else None,
                    collected_frames=n, collected_episodes=int(len(np.unique(data["episode_id"]))),
                    completed_episodes=int(np.count_nonzero(data["episode_end"])),
                    context_warmup=8, max_block_length=64,
                    latent_sampling="random_sample; reset seed per block, shared across stages/models",
                    background="Uncorrupted shared rollout; replace igloo throughout each context block",
                    uncertainty_scope="Evaluation contexts of two fixed checkpoints, not training-seed variation")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    analyze_values(output, args.bootstrap, args.seed, args.bootstrap_unit)


if __name__ == "__main__":
    main()
