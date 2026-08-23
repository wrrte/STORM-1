import numpy as np
import cv2
import torch
import os
import argparse
from scipy.stats import spearmanr, kendalltau
from utils import load_config
conf = load_config("config_files/STORM.yaml")
from train import build_world_model, build_agent

def get_igloo_stage(ram_77_val):
    if ram_77_val == 255:
        return 0
    else:
        return ram_77_val + 1

def load_models(run_name, step, env_name):
    import gymnasium
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

def collect_episodes_data(env_name, world_model, agent, target_frames=20000):
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
    
    obs, info = env.reset()
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
        all_ram.append(ram)
        
        # obs_tensor needs to be (1, 1, 3, 64, 64) float32 in [0, 1]
        obs_tensor = torch.Tensor(obs_np).cuda().unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 2, 3) / 255.0
        context_obs.append(obs_tensor)
        context_action.append([action])
        
        obs, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            obs, info = env.reset()
            context_obs.clear()
            context_action.clear()
            
    print(f"데이터 수집 완료!")
    return np.array(all_obs), np.array(all_action), np.array(all_ram)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="Run name inside ckpt/ directory")
    parser.add_argument("--frames", type=int, default=20000, help="Number of frames to collect for buffer")
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print(f"[{args.run_name}] 모델 평가 시작")
    print(f"{'='*50}")
    
    world_model, agent = load_models(args.run_name, 100000, "ALE/Frostbite-v5")
    
    # 1. 템플릿 이미지를 폴더에서 로드
    templates = {}
    template_dir = "frostbite_igloo_templates"
    
    if not os.path.exists(template_dir):
        print(f"'{template_dir}' 폴더가 없습니다! 먼저 'python extract_igloo_templates.py'를 실행해주세요.")
        return
        
    for stg in range(17):
        path = os.path.join(template_dir, f"stage_{stg:02d}.png")
        if not os.path.exists(path):
            print(f"오류: {path} 파일을 찾을 수 없습니다!")
            return
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        templates[stg] = img_rgb
        
    # 2. 동적 데이터 수집 (Replay Buffer 생성)
    obs, action, ram = collect_episodes_data("ALE/Frostbite-v5", world_model, agent, target_frames=args.frames)
    
    if len(action.shape) == 2 and action.shape[1] == 1:
        action = action.squeeze(-1)
        
    stages = np.array([get_igloo_stage(r[77]) for r in ram])
    
    # 3. 동일한 이글루 단계를 유지하는 블록 스캔 (길이 9 이상, 최대 64)
    valid_blocks = []
    i = 0
    max_seq_len = 64 # Transformer max length
    
    while i < len(obs):
        stage = stages[i]
        j = i + 1
        while j < len(obs) and stages[j] == stage and (j - i) < max_seq_len:
            j += 1
        
        length = j - i
        if length >= 9:
            valid_blocks.append((i, j))
        
        i = j
            
    print(f"\n총 {len(valid_blocks)}개의 연속된 유효 블록(길이 9 이상) 발견.")
    if len(valid_blocks) == 0:
        print("유효한 구간이 없습니다. --frames 값을 더 크게 설정하여 다시 시도해보세요.")
        return
    
    # 4. 가치 함수 평가 (캐시 재사용 최적화)
    all_values = [] # 각 유효 프레임마다의 17개 가치를 저장
    
    from tqdm import tqdm
    print("\n가치 함수(Value Function) 평가 중...")
    for start, end in tqdm(valid_blocks, desc="Evaluating blocks"):
        block_length = end - start
        num_eval_frames = block_length - 8
        
        base_obs = obs[start:end].copy()
        base_action = action[start:end-1].copy()
        
        base_action_t = torch.from_numpy(base_action).long().cuda().unsqueeze(0) # (1, block_length-1)
        
        # 특정 블록 내에서 17개의 평가를 진행
        # 구조: block_values[frame_idx][stg]
        block_values = [[] for _ in range(num_eval_frames)]
        
        for stg in range(17):
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
    
    # 5. 평가 결과 도출
    ideal_ranks = np.arange(17)
    
    spearman_scores = []
    kendall_scores = []
    
    for i in range(len(all_values)):
        rho, _ = spearmanr(ideal_ranks, all_values[i])
        tau, _ = kendalltau(ideal_ranks, all_values[i])
        if not np.isnan(rho): spearman_scores.append(rho)
        if not np.isnan(tau): kendall_scores.append(tau)
        
    avg_spearman = np.mean(spearman_scores)
    avg_kendall = np.mean(kendall_scores)
    
    global_avg_values = np.mean(all_values, axis=0)
    global_spearman, _ = spearmanr(ideal_ranks, global_avg_values)
    global_kendall, _ = kendalltau(ideal_ranks, global_avg_values)
    
    print(f"\n{'='*50}")
    print(f"[{args.run_name}] 최종 평가 결과")
    print(f"{'='*50}")
    print(f"[Metric 1: Per-Sequence 평균]")
    print(f"  - Spearman's rho : {avg_spearman:.4f}")
    print(f"  - Kendall Tau    : {avg_kendall:.4f}")
    print(f"\n[Metric 2: Global Average Profile]")
    print(f"  - Spearman's rho : {global_spearman:.4f}")
    print(f"  - Kendall Tau    : {global_kendall:.4f}")
    print(f"{'='*50}")
    
    print("\n[단계별 Global Average Value]")
    for i, v in enumerate(global_avg_values):
        print(f"  Stage {i:02d} : {v:.4f}")

if __name__ == "__main__":
    main()
