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

from utils import seed_np_torch, Logger, load_config
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel, MSELoss
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval import RetrievalContextManager


def build_single_env(env_name, image_size, seed):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.SeedEnvWrapper(env, seed=seed)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfo(env)
    return env


def build_vec_env(env_name, image_size, num_envs, seed):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size, seed)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModel, batch_size, demonstration_batch_size, batch_length, logger, agent=None, retrieval_manager=None, imagine_context_length=8, is_warmup=False):
    obs, action, reward, termination, base_indexes, base_envs = replay_buffer.sample(batch_size, demonstration_batch_size, batch_length)
    latent, dist_feat = world_model.update(obs, action, reward, termination, logger=logger)

    if agent is not None and retrieval_manager is not None and retrieval_manager.enabled:
        with torch.no_grad():
            agent_state = torch.cat([latent, dist_feat], dim=-1)
            v_t = agent.value(agent_state).squeeze(-1) # shape: (batch_size, batch_length)
            gamma = getattr(agent, "gamma", 0.995)
            num_trig = retrieval_manager.add_batch_transitions(v_t, reward, termination, gamma, base_indexes, base_envs, replay_buffer.max_length, skip_len=imagine_context_length, is_warmup=is_warmup)
            return num_trig
    return 0


@torch.no_grad()
def world_model_imagine_data(replay_buffer: ReplayBuffer,
                             world_model: WorldModel, agent: agents.ActorCriticAgent,
                             retrieval_manager,
                             imagine_batch_size, imagine_demonstration_batch_size,
                             imagine_context_length, imagine_batch_length,
                             log_video, logger, is_warmup=False):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()

    retrieved_count = 0
    lazy_hit_rate = 0.0
    if retrieval_manager is not None and retrieval_manager.enabled and not is_warmup:
        cfg = retrieval_manager.config
        ret_obs, ret_action, candidates_before_max, avg_hit_rate, retrieved_weights, num_valid_anchors = retrieval_manager.retrieve_contexts(
            replay_buffer, world_model, 
            max_anchors=cfg.get("max_anchors", 10) if hasattr(cfg, "get") else getattr(cfg, "max_anchors", 10),
            multiplier=cfg.get("multiplier", 5) if hasattr(cfg, "get") else getattr(cfg, "multiplier", 5),
            target=cfg.get("target", 5) if hasattr(cfg, "get") else getattr(cfg, "target", 5),
            max_contexts=cfg.get("max_contexts", 256) if hasattr(cfg, "get") else getattr(cfg, "max_contexts", 256)
        )
        if ret_obs is not None:
            retrieved_count = ret_obs.shape[0]
            logger.log("Retrieval/candidates_before_max", candidates_before_max)
            logger.log("Retrieval/lazy_rebuild_hit_rate", avg_hit_rate)
            lazy_hit_rate = avg_hit_rate
            
    if retrieved_count > 0:
        random_batch_size = imagine_batch_size - num_valid_anchors
    else:
        random_batch_size = imagine_batch_size
            
    sample_obs, sample_action, sample_reward, sample_termination, _, _ = replay_buffer.sample(
        random_batch_size, imagine_demonstration_batch_size, imagine_context_length)
        
    batch_weights = torch.ones(random_batch_size, device="cuda", dtype=torch.float32)
        
    if retrieved_count > 0:
        sample_obs = torch.cat([sample_obs, ret_obs], dim=0)
        sample_action = torch.cat([sample_action, ret_action], dim=0)
        
        ret_weights = torch.tensor(retrieved_weights, device="cuda", dtype=torch.float32)
        batch_weights = torch.cat([batch_weights, ret_weights], dim=0)
        
        logger.log("Retrieval/retrieved_contexts", retrieved_count)
        logger.log("Retrieval/active_anchors_queue", len(retrieval_manager.active_anchors))
        
        if len(retrieval_manager.hash_memory) > 0:
            sizes = [len(b) for b in retrieval_manager.hash_memory.values()]
            avg_size = sum(sizes) / len(sizes)
            logger.log("Retrieval/avg_bucket_size", avg_size)
            logger.log("Retrieval/num_active_buckets", len(sizes))
            
            # Record bucket distribution occasionally (e.g., if sizes has items)
            # wandb histograms need a list of values
            logger.log("Retrieval/bucket_size_hist", np.array(sizes))
                
    final_batch_size = sample_obs.shape[0]

    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent, sample_obs, sample_action,
        imagine_batch_size=final_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger
    )
    return latent, action, None, None, reward_hat, termination_hat, lazy_hit_rate, batch_weights


def joint_train_world_model_agent(env_name, max_steps, num_envs, image_size,
                                  replay_buffer: ReplayBuffer,
                                  world_model: WorldModel, agent: agents.ActorCriticAgent,
                                  train_dynamics_every_steps, train_agent_every_steps,
                                  batch_size, demonstration_batch_size, batch_length,
                                  imagine_batch_size, imagine_demonstration_batch_size,
                                  imagine_context_length, imagine_batch_length,
                                  save_every_steps, seed, logger):
    # create ckpt dir
    os.makedirs(f"ckpt/{args.n}", exist_ok=True)

    # build vec env, not useful in the Atari100k setting
    # but when the max_steps is large, you can use parallel envs to speed up
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs, seed=seed)
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)

    # reset envs and variables
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)
    
    # retrieval setup
    retrieval_config = getattr(conf.JointTrainAgent, 'Retrieval', {})
    if hasattr(retrieval_config, 'update'): # if it's a dict or omegaconf
        retrieval_config['context_length'] = imagine_context_length
    else: # if it's omegaconf DictConfig, we can set attribute
        setattr(retrieval_config, 'context_length', imagine_context_length)
        
    latent_dim = 32 * 32 # CategoricalDim * ClassDim for hashing only the single-frame latent
    retrieval_manager = RetrievalContextManager(num_envs=num_envs, config=retrieval_config, latent_dim=latent_dim)
    is_first_step = np.ones(num_envs, dtype=bool)
    
    # global rebuild tracking
    last_rebuild_step = -getattr(retrieval_config, "global_rebuild_cooldown", 2000)

    # sample and train
    for total_steps in tqdm(range(max_steps//num_envs)):
        # sample part >>>
        if replay_buffer.ready():
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.action_space.sample()
                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).cuda()
                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    agent_state = torch.cat([prior_flattened_sample, last_dist_feat], dim=-1)
                    action = agent.sample_as_env_action(
                        agent_state,
                        greedy=False
                    )
                    current_latent_for_hash = world_model.encode_obs(
                        torch.cat(list(context_obs), dim=1)[:, -1:], 
                        sample_mode=getattr(retrieval_config, "hash_sample_mode", "probs")
                    ).squeeze(1)

            context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
            context_action.append(action)
        else:
            action = vec_env.action_space.sample()
            agent_state = None
            current_latent_for_hash = None

        obs, reward, done, truncated, info = vec_env.step(action)
        replay_buffer.append(current_obs, action, reward, np.logical_or(done, info["life_loss"]))
        
        if replay_buffer.ready() and current_latent_for_hash is not None:
            retrieval_manager.add_transition(
                pointer=replay_buffer.last_pointer,
                env_idx=-1,
                latent_b=current_latent_for_hash
            )
        
        is_first_step = done

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    logger.log(f"sample/{env_name}_reward", sum_reward[i])
                    logger.log(f"sample/{env_name}_episode_steps", current_info["episode_frame_number"][i]//4)  # framskip=4
                    logger.log("replay_buffer/length", len(replay_buffer))
                    sum_reward[i] = 0

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        # <<< sample part

        # train world model part >>>
        if replay_buffer.ready() and total_steps % (train_dynamics_every_steps//num_envs) == 0:
            is_retrieval_warmup = total_steps * num_envs < getattr(retrieval_config, "warmup_steps", 5000)

            num_trig = train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=batch_size,
                demonstration_batch_size=demonstration_batch_size,
                batch_length=batch_length,
                logger=logger,
                agent=agent,
                retrieval_manager=retrieval_manager,
                imagine_context_length=imagine_context_length,
                is_warmup=is_retrieval_warmup
            )
            
            logger.log("Retrieval/triggered_anchors_step", num_trig)
        # <<< train world model part

        # train agent part >>>
        if replay_buffer.ready() and total_steps % (train_agent_every_steps//num_envs) == 0 and total_steps*num_envs >= 0:
            if total_steps % (save_every_steps//num_envs) == 0:
                log_video = True
                
                video_columns = 5
                video_temporal_length = 5
                num_videos = video_columns * video_temporal_length
                openloop_obs, openloop_action, _, _, _, _ = replay_buffer.sample(
                    num_videos, 0, imagine_context_length + imagine_batch_length)
                
                world_model.log_openloop_video(
                    openloop_obs, openloop_action, imagine_context_length, imagine_batch_length, logger, video_columns=video_columns)
            else:
                log_video = False

            is_retrieval_warmup = total_steps * num_envs < getattr(retrieval_config, "warmup_steps", 5000)

            imagine_latent, agent_action, agent_logprob, agent_value, imagine_reward, imagine_termination, lazy_hit_rate, batch_weights = world_model_imagine_data(
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                retrieval_manager=retrieval_manager,
                imagine_batch_size=imagine_batch_size,
                imagine_demonstration_batch_size=imagine_demonstration_batch_size,
                imagine_context_length=imagine_context_length,
                imagine_batch_length=imagine_batch_length,
                log_video=log_video,
                logger=logger,
                is_warmup=is_retrieval_warmup
            )
            
            # Global Rebuild Check
            if retrieval_manager is not None and retrieval_manager.enabled:
                gr_enabled = getattr(retrieval_config, "global_rebuild_enable", True)
                gr_threshold = getattr(retrieval_config, "global_rebuild_threshold", 0.2)
                gr_cooldown = getattr(retrieval_config, "global_rebuild_cooldown", 2000)
                
                warmup_steps = getattr(retrieval_config, "warmup_steps", 15000)
                buffer_warmup = getattr(conf.JointTrainAgent, "BufferWarmUp", 1024)
                
                train_freq = max(1, train_agent_every_steps // num_envs)
                next_train_step = total_steps + train_freq
                
                is_just_before_warmup_end = (
                    is_retrieval_warmup 
                    and warmup_steps > buffer_warmup 
                    and (next_train_step * num_envs >= warmup_steps)
                )
                
                if is_retrieval_warmup and not is_just_before_warmup_end:
                    logger.log("Retrieval/global_rebuild_triggered", 0.0)
                else:
                    if is_just_before_warmup_end or (gr_enabled and lazy_hit_rate < gr_threshold and total_steps - last_rebuild_step >= gr_cooldown):
                        retrieval_manager.rebuild_all_hash_buckets(replay_buffer, world_model, chunk_size=1024)
                        last_rebuild_step = total_steps
                        logger.log("Retrieval/global_rebuild_triggered", 1.0)
                    else:
                        logger.log("Retrieval/global_rebuild_triggered", 0.0)

            agent.update(
                latent=imagine_latent,
                action=agent_action,
                old_logprob=agent_logprob,
                old_value=agent_value,
                reward=imagine_reward,
                termination=imagine_termination,
                logger=logger,
                weights=batch_weights
            )
        # <<< train agent part

        # save model per episode
        if total_steps % (save_every_steps//num_envs) == 0:
            print(colorama.Fore.GREEN + f"Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), f"ckpt/{args.n}/world_model_{total_steps}.pth")
            torch.save(agent.state_dict(), f"ckpt/{args.n}/agent_{total_steps}.pth")

        # flush all buffered wandb metrics for this step
        logger.flush_wandb()


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
        feat_dim=32*32+conf.Models.WorldModel.TransformerHiddenDim,
        num_layers=conf.Models.Agent.NumLayers,
        hidden_dim=conf.Models.Agent.HiddenDim,
        action_dim=action_dim,
        gamma=conf.Models.Agent.Gamma,
        lambd=conf.Models.Agent.Lambda,
        entropy_coef=conf.Models.Agent.EntropyCoef,
    ).cuda()


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-trajectory_path", type=str, required=True)
    args, extra_args = parser.parse_known_args()
    conf = load_config(args.config_path)
    if extra_args:
        conf.defrost()
        conf.merge_from_list(extra_args)
        conf.freeze()
    print(colorama.Fore.RED + str(args) + " extra: " + str(extra_args) + colorama.Style.RESET_ALL)

    if hasattr(conf, "JointTrainAgent") and hasattr(conf.JointTrainAgent, "Retrieval"):
        args.n += "_O" if conf.JointTrainAgent.Retrieval.enable else "_X"


    # set seed
    seed_np_torch(seed=args.seed)
    # tensorboard writer
    logger = Logger(path=f"runs/{args.n}", config=conf)
    # copy config file
    shutil.copy(args.config_path, f"runs/{args.n}/config.yaml")

    # distinguish between tasks, other debugging options are removed for simplicity
    if conf.Task == "JointTrainAgent":
        # getting action_dim with dummy env
        dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize, seed=0)
        action_dim = dummy_env.action_space.n

        # build world model and agent
        world_model = build_world_model(conf, action_dim)
        agent = build_agent(conf, action_dim)

        # build replay buffer
        replay_buffer = ReplayBuffer(
            obs_shape=(conf.BasicSettings.ImageSize, conf.BasicSettings.ImageSize, 3),
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_length=conf.JointTrainAgent.BufferMaxLength,
            warmup_length=conf.JointTrainAgent.BufferWarmUp,
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU
        )

        # judge whether to load demonstration trajectory
        if conf.JointTrainAgent.UseDemonstration:
            print(colorama.Fore.MAGENTA + f"loading demonstration trajectory from {args.trajectory_path}" + colorama.Style.RESET_ALL)
            replay_buffer.load_trajectory(path=args.trajectory_path)

        # train
        joint_train_world_model_agent(
            env_name=args.env_name,
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            image_size=conf.BasicSettings.ImageSize,
            replay_buffer=replay_buffer,
            world_model=world_model,
            agent=agent,
            train_dynamics_every_steps=conf.JointTrainAgent.TrainDynamicsEverySteps,
            train_agent_every_steps=conf.JointTrainAgent.TrainAgentEverySteps,
            batch_size=conf.JointTrainAgent.BatchSize,
            demonstration_batch_size=conf.JointTrainAgent.DemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
            batch_length=conf.JointTrainAgent.BatchLength,
            imagine_batch_size=conf.JointTrainAgent.ImagineBatchSize,
            imagine_demonstration_batch_size=conf.JointTrainAgent.ImagineDemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
            imagine_context_length=conf.JointTrainAgent.ImagineContextLength,
            imagine_batch_length=conf.JointTrainAgent.ImagineBatchLength,
            save_every_steps=conf.JointTrainAgent.SaveEverySteps,
            seed=args.seed,
            logger=logger
        )

        print(colorama.Fore.GREEN + f"Evaluating the trained model before finishing..." + colorama.Style.RESET_ALL)
        import eval
        episode_avg_return, individual_returns = eval.eval_episodes(
            num_episode=20,
            env_name=args.env_name,
            num_envs=5,
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            image_size=conf.BasicSettings.ImageSize,
            world_model=world_model,
            agent=agent
        )

        env_base = args.env_name.split('/')[1].split('-')[0]
        baseline_score_avg = 0.0
        try:
            with open("results/storm.json", "r") as f:
                storm_results = json.load(f)
                if env_base in storm_results:
                    baseline_score_avg = float(np.mean(storm_results[env_base]))
        except Exception as e:
            print(f"Failed to load baseline scores for {env_base}: {e}")

        logger.log("eval/baseline_score_avg", baseline_score_avg)
        logger.log("eval/episode_avg_return", episode_avg_return)
        for i, ret in enumerate(individual_returns):
            logger.log(f"eval/episode_return_{i}", ret)
        logger.flush_wandb()

    else:
        raise NotImplementedError(f"Task {conf.Task} not implemented")
