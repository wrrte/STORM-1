import gymnasium
import argparse
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import copy
import colorama
import random
import json
import shutil
import pickle
import os
import csv

from utils import seed_np_torch, Logger, load_config, configure_performance
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.functions_losses import symexp
from sub_models.precision import evaluation_context, evaluation_precisions
from sub_models.world_models import WorldModel, MSELoss


def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))
    return img


def build_single_env(env_name, image_size):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    return env


def build_vec_env(env_name, image_size, num_envs):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def eval_episodes(num_episode, env_name, max_steps, num_envs, image_size,
                  world_model: WorldModel, agent: agents.ActorCriticAgent,
                  precision="auto", seed=None):
    with evaluation_context(world_model, agent, precision, seed):
        return _eval_episodes(num_episode, env_name, max_steps, num_envs, image_size,
                              world_model, agent, seed)


def _eval_episodes(num_episode, env_name, max_steps, num_envs, image_size,
                   world_model, agent, seed):
    """Evaluate exactly one complete episode per environment."""
    if num_episode <= 0 or num_envs != num_episode:
        raise ValueError("First-episode evaluation requires num_envs == num_episode > 0")
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs)
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)
    sum_reward = np.zeros(num_envs)
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    completed = np.zeros(num_envs, dtype=bool)
    final_rewards = np.zeros(num_envs)
    try:
        if seed is None:
            current_obs, current_info = vec_env.reset()
        else:
            # Distinct environment seeds, identical lists for BF16 and FP32.
            vec_env.action_space.seed(seed)
            current_obs, current_info = vec_env.reset(seed=[seed + i for i in range(num_envs)])
        while not completed.all():
            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.action_space.sample()
                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).cuda()
                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False
                    )

            context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
            context_action.append(action)

            obs, reward, done, truncated, info = vec_env.step(action)
            # Include terminal rewards and ignore auto-reset episodes in finished slots.
            sum_reward[~completed] += reward[~completed]
            newly_completed = np.logical_or(done, truncated) & ~completed
            final_rewards[newly_completed] = sum_reward[newly_completed]
            completed |= newly_completed
            current_obs = obs
            current_info = info

        mean_reward = np.mean(final_rewards)
        print("Mean reward: " + colorama.Fore.YELLOW + f"{mean_reward}" + colorama.Style.RESET_ALL)
        return mean_reward, final_rewards.tolist()
    finally:
        vec_env.close()


def compare_precisions(precision="both", seed=0, **kwargs):
    """Evaluate the same weights and initial RNG states for each precision."""
    results = {}
    for mode in evaluation_precisions(precision):
        print(f"Evaluation precision: {mode}, seed: {seed}")
        results[mode] = eval_episodes(**kwargs, precision=mode, seed=seed)
    return results


def evaluate_and_log(logger, precision="auto", seed=0, **kwargs):
    results = compare_precisions(precision=precision, seed=seed, **kwargs)
    logger.log("eval/seed", seed)
    for mode, (mean_reward, returns) in results.items():
        if mode != "auto":
            logger.log(f"eval/{mode}/episode_avg_return", mean_reward)
            for i, reward in enumerate(returns):
                logger.log(f"eval/{mode}/episode_return_{i}", reward)
    if precision == "both":
        logger.log("eval/fp32_minus_bf16", results["fp32"][0] - results["bf16"][0])
    # Keep existing dashboards on the first arm (BF16 when comparing both).
    mean_reward, returns = next(iter(results.values()))
    logger.log("eval/episode_avg_return", mean_reward)
    for i, reward in enumerate(returns):
        logger.log(f"eval/episode_return_{i}", reward)
    return mean_reward, returns


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-run_name", type=str, required=True)
    parser.add_argument("-eval_all", action="store_true", help="Evaluate all checkpoints")
    parser.add_argument("-reverse", action="store_true", help="Evaluate checkpoints in reverse order (latest first)")
    parser.add_argument("-precision", "--precision", choices=("auto", "bf16", "fp32", "both"),
                        default=None, help="Evaluation precision (default: JointTrainAgent.EvalPrecision)")
    parser.add_argument("-seed", "--seed", type=int, default=None, help="Evaluation RNG and first environment seed")
    parser.add_argument("-episodes", "--episodes", type=int, default=None, help="One complete episode per environment")
    parser.add_argument("-step", "--step", type=int, default=None, help="Evaluate only this checkpoint step")
    args = parser.parse_args()
    conf = load_config(args.config_path)
    precision = args.precision if args.precision is not None else conf.JointTrainAgent.EvalPrecision
    eval_seed = args.seed if args.seed is not None else conf.JointTrainAgent.EvalSeed
    episodes = args.episodes if args.episodes is not None else conf.JointTrainAgent.EvalEpisodes
    if episodes <= 0 or not (0 <= eval_seed <= 2**32 - 1):
        parser.error("episodes must be positive and seed must be between 0 and 2**32 - 1")
    if args.step is not None and args.step < 0:
        parser.error("step must be nonnegative")
    evaluation_precisions(precision)
    configure_performance(conf)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)
    # print(colorama.Fore.RED + str(conf) + colorama.Style.RESET_ALL)

    # set seed
    seed_np_torch(seed=eval_seed)

    # build and load model/agent
    import train
    dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize)
    action_dim = dummy_env.action_space.n
    dummy_env.close()
    world_model = train.build_world_model(conf, action_dim)
    agent = train.build_agent(conf, action_dim)
    root_path = f"ckpt/{args.run_name}"

    import glob
    pathes = glob.glob(f"{root_path}/world_model_[0-9]*.pth")
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    if args.step is not None:
        steps = [args.step]
    elif not args.eval_all:
        steps = steps[-1:]
    if not steps:
        parser.error(f"No numeric checkpoints found in {root_path}")
    for step in steps:
        for name in ("world_model", "agent"):
            path = f"{root_path}/{name}_{step}.pth"
            if not os.path.isfile(path):
                parser.error(f"Checkpoint not found: {path}")
    
    if args.reverse:
        steps.reverse()

    print(steps)

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = "" if precision == "auto" else f"_{precision}_seed{eval_seed}"
    csv_path = f"eval_result/{args.run_name}{suffix}_{timestamp}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    episode_csv_path = csv_path.removesuffix(".csv") + "_episodes.csv"
    print(colorama.Fore.GREEN + f"Results will be saved to: {csv_path}" + colorama.Style.RESET_ALL)

    with open(csv_path, "w") as fout:
        writer = csv.writer(fout)
        writer.writerow(["step", " episode_avg_return"] if precision == "auto"
                        else ["step", "precision", "seed", "episode_avg_return"])
    with open(episode_csv_path, "w", newline="") as fout:
        csv.writer(fout).writerow(["step", "precision", "seed", "episode", "env_seed", "episode_return"])

    results = []
    for step in tqdm(steps):
        world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth"))
        agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))
        # # eval
        comparison = compare_precisions(
            precision=precision,
            seed=eval_seed,
            num_episode=episodes,
            env_name=args.env_name,
            num_envs=episodes,
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            image_size=conf.BasicSettings.ImageSize,
            world_model=world_model,
            agent=agent
        )
        with open(csv_path, "a", newline="") as fout, open(episode_csv_path, "a", newline="") as episodes_out:
            for mode, (episode_avg_return, returns) in comparison.items():
                results.append([step, mode, episode_avg_return])
                csv.writer(fout).writerow([step, episode_avg_return] if precision == "auto"
                                         else [step, mode, eval_seed, episode_avg_return])
                csv.writer(episodes_out).writerows(
                    [step, mode, eval_seed, i, eval_seed + i, reward]
                    for i, reward in enumerate(returns))
        if precision == "both":
            print(f"FP32 - BF16 at step {step}: {comparison['fp32'][0] - comparison['bf16'][0]}")
    print(f"Saved means to {csv_path}; individual episodes to {episode_csv_path}")
