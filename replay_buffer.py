import numpy as np
import random
import unittest
import torch
from einops import rearrange
import copy
import pickle


class ReplayBuffer():
    def __init__(self, obs_shape, num_envs, max_length=int(1E6), warmup_length=50000, store_on_gpu=False) -> None:
        self.store_on_gpu = store_on_gpu
        if store_on_gpu:
            self.obs_buffer = torch.empty((max_length//num_envs, num_envs, *obs_shape), dtype=torch.uint8, device="cuda", requires_grad=False)
            self.action_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device="cuda", requires_grad=False)
            self.reward_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device="cuda", requires_grad=False)
            self.termination_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device="cuda", requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length//num_envs, num_envs, *obs_shape), dtype=np.uint8)
            self.action_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            self.reward_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            self.termination_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)

        self.length = 0
        self.num_envs = num_envs
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None

    def load_trajectory(self, path):
        buffer = pickle.load(open(path, "rb"))
        if self.store_on_gpu:
            self.external_buffer = {name: torch.from_numpy(buffer[name]).to("cuda") for name in buffer}
        else:
            self.external_buffer = buffer
        self.external_buffer_length = self.external_buffer["obs"].shape[0]

    def sample_external(self, batch_size, batch_length, to_device="cuda"):
        indexes = np.random.randint(0, self.external_buffer_length+1-batch_length, size=batch_size)
        if self.store_on_gpu:
            obs = torch.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
            action = torch.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
            reward = torch.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
            termination = torch.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
        else:
            obs = np.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
            action = np.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
            reward = np.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
            termination = np.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
        return obs, action, reward, termination

    def ready(self):
        return self.length * self.num_envs > self.warmup_length

    @torch.no_grad()
    def sample(self, batch_size, external_batch_size, batch_length, to_device="cuda"):
        if self.store_on_gpu:
            obs, action, reward, termination = [], [], [], []
            base_indexes, base_envs = [], []
            if batch_size > 0:
                batch_per_env = batch_size // self.num_envs
                indexes = np.random.randint(0, self.length + 1 - batch_length, size=(self.num_envs, batch_per_env))
                env_idx = np.arange(self.num_envs)[:, None]
                
                p_grid = indexes[..., None] + np.arange(batch_length)
                e_grid = np.broadcast_to(env_idx[..., None], p_grid.shape)
                
                obs_chunk = self.obs_buffer[p_grid, e_grid]
                obs.append(obs_chunk.reshape(-1, batch_length, *self.obs_buffer.shape[2:]))
                
                action_chunk = self.action_buffer[p_grid, e_grid]
                action.append(action_chunk.reshape(-1, batch_length, *self.action_buffer.shape[2:]))
                
                reward_chunk = self.reward_buffer[p_grid, e_grid]
                reward.append(reward_chunk.reshape(-1, batch_length, *self.reward_buffer.shape[2:]))
                
                term_chunk = self.termination_buffer[p_grid, e_grid]
                termination.append(term_chunk.reshape(-1, batch_length, *self.termination_buffer.shape[2:]))
                
                base_indexes.append(indexes.flatten())
                base_envs.append(np.broadcast_to(env_idx, indexes.shape).flatten())

            if self.external_buffer_length is not None and external_batch_size > 0:
                external_obs, external_action, external_reward, external_termination = self.sample_external(
                    external_batch_size, batch_length, to_device)
                obs.append(external_obs)
                action.append(external_action)
                reward.append(external_reward)
                termination.append(external_termination)
                base_indexes.append(np.full(external_batch_size, -1))
                base_envs.append(np.full(external_batch_size, -1))

            obs = torch.cat(obs, dim=0).float() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.cat(action, dim=0)
            reward = torch.cat(reward, dim=0)
            termination = torch.cat(termination, dim=0)
            base_indexes = np.concatenate(base_indexes, axis=0)
            base_envs = np.concatenate(base_envs, axis=0)
        else:
            obs, action, reward, termination = [], [], [], []
            base_indexes, base_envs = [], []
            if batch_size > 0:
                batch_per_env = batch_size // self.num_envs
                indexes = np.random.randint(0, self.length + 1 - batch_length, size=(self.num_envs, batch_per_env))
                env_idx = np.arange(self.num_envs)[:, None]
                
                p_grid = indexes[..., None] + np.arange(batch_length)
                e_grid = np.broadcast_to(env_idx[..., None], p_grid.shape)
                
                obs_chunk = self.obs_buffer[p_grid, e_grid]
                obs.append(obs_chunk.reshape(-1, batch_length, *self.obs_buffer.shape[2:]))
                
                action_chunk = self.action_buffer[p_grid, e_grid]
                action.append(action_chunk.reshape(-1, batch_length, *self.action_buffer.shape[2:]))
                
                reward_chunk = self.reward_buffer[p_grid, e_grid]
                reward.append(reward_chunk.reshape(-1, batch_length, *self.reward_buffer.shape[2:]))
                
                term_chunk = self.termination_buffer[p_grid, e_grid]
                termination.append(term_chunk.reshape(-1, batch_length, *self.termination_buffer.shape[2:]))
                
                base_indexes.append(indexes.flatten())
                base_envs.append(np.broadcast_to(env_idx, indexes.shape).flatten())

            if self.external_buffer_length is not None and external_batch_size > 0:
                external_obs, external_action, external_reward, external_termination = self.sample_external(
                    external_batch_size, batch_length, to_device)
                obs.append(external_obs)
                action.append(external_action)
                reward.append(external_reward)
                termination.append(external_termination)
                base_indexes.append(np.full(external_batch_size, -1))
                base_envs.append(np.full(external_batch_size, -1))

            obs = torch.from_numpy(np.concatenate(obs, axis=0)).float().cuda() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.from_numpy(np.concatenate(action, axis=0)).cuda()
            reward = torch.from_numpy(np.concatenate(reward, axis=0)).cuda()
            termination = torch.from_numpy(np.concatenate(termination, axis=0)).cuda()
            base_indexes = np.concatenate(base_indexes, axis=0)
            base_envs = np.concatenate(base_envs, axis=0)

        return obs, action, reward, termination, base_indexes, base_envs

    def append(self, obs, action, reward, termination):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % (self.max_length//self.num_envs)
        if self.store_on_gpu:
            self.obs_buffer[self.last_pointer] = obs if isinstance(obs, torch.Tensor) else torch.from_numpy(obs)
            self.action_buffer[self.last_pointer] = action if isinstance(action, torch.Tensor) else torch.from_numpy(action)
            self.reward_buffer[self.last_pointer] = reward if isinstance(reward, torch.Tensor) else torch.from_numpy(reward)
            self.termination_buffer[self.last_pointer] = termination if isinstance(termination, torch.Tensor) else torch.from_numpy(termination)
        else:
            self.obs_buffer[self.last_pointer] = obs
            self.action_buffer[self.last_pointer] = action
            self.reward_buffer[self.last_pointer] = reward
            self.termination_buffer[self.last_pointer] = termination

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length * self.num_envs
