"""Evaluation-only RAM-aligned CPU replay, with the original retrieval API."""
from pathlib import Path

import numpy as np


class EvaluationReplay:
    def __init__(self, capacity, image_size=64, directory=None):
        self.capacity = self.max_length = capacity
        self.num_envs, self.store_on_gpu = 1, False
        self.length, self.last_pointer = 0, -1
        shapes = dict(obs=((capacity, 1, image_size, image_size, 3), np.uint8),
                      ram=((capacity, 128), np.uint8), action=((capacity, 1), np.int64),
                      reward=((capacity, 1), np.float32), termination=((capacity, 1), np.float32),
                      global_step=((capacity,), np.int64), episode=((capacity,), np.int64),
                      episode_step=((capacity,), np.int64), life_id=((capacity,), np.int64))
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
        for name, (shape, dtype) in shapes.items():
            array = (np.lib.format.open_memmap(Path(directory) / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
                     if directory else np.empty(shape, dtype=dtype))
            if name in ("global_step", "episode", "episode_step", "life_id"):
                array.fill(-1)
            setattr(self, name + "_buffer", array)

    def append(self, obs, ram, action, reward, termination, global_step, episode, episode_step, life_id):
        p = (self.last_pointer + 1) % self.capacity
        for name, value in locals().copy().items():
            if name not in ("self", "p"):
                getattr(self, name + "_buffer")[p] = value
        self.last_pointer, self.length = p, min(self.length + 1, self.capacity)
        return p

    def position(self, pointer):
        oldest = (self.last_pointer + 1) % self.capacity if self.length == self.capacity else 0
        return (pointer - oldest) % self.capacity

    def is_valid_context(self, pointer, env, length):
        if env != 0 or not 0 <= pointer < self.capacity:
            return False
        position = self.position(pointer)
        if position >= self.length or position + 1 < length:
            return False
        indexes = (pointer - np.arange(1, length)) % self.capacity
        return bool(np.all(self.termination_buffer[indexes, 0] <= .5))

    def valid_endpoints(self, length):
        oldest = (self.last_pointer + 1) % self.capacity if self.length == self.capacity else 0
        pointers = (oldest + np.arange(max(0, length - 1), self.length)) % self.capacity
        valid = np.ones(len(pointers), bool)
        for lag in range(1, length):
            valid &= self.termination_buffer[(pointers - lag) % self.capacity, 0] <= .5
        return pointers[valid]

    def gather_retrieval_frames(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return np.asarray(self.obs_buffer[indices[:, 0], indices[:, 1]])

    def gather_retrieval_contexts(self, indices, context_length):
        indices = np.asarray(indices, dtype=np.int64)
        times = (indices[None, :, 0] - np.arange(context_length - 1, -1, -1)[:, None]) % self.capacity
        return (np.asarray(self.obs_buffer[times, indices[None, :, 1]]),
                np.asarray(self.action_buffer[times, indices[None, :, 1]]))

    def sample_random(self, count, anchor, context_length, rng):
        pool = self.valid_endpoints(context_length)
        pool = pool[pool != anchor]
        if len(pool) < count:
            raise ValueError("Not enough valid random endpoints to match retrieval count")
        return rng.choice(pool, count, replace=False).astype(int).tolist()

    def flush(self):
        for name, value in vars(self).items():
            if name.endswith("_buffer") and hasattr(value, "flush"):
                value.flush()
