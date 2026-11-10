import torch
import random
import numpy as np
import copy
import math
from collections import deque
# Evaluation-only enable parsing; both retrieval classes below are verbatim.
RETRIEVAL_EXPERIMENTS = {}
def parse_retrieval_mode(value, allow_experiments=False):
    if value is not True:
        raise ValueError("Evaluation snapshot requires enable=True")
    return True


class FastHashBucket:
    """O(N) 병목을 제거하고 완전한 O(1) 연산을 지원하는 딕셔너리 기반 해시 버킷"""
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = [] # 랜덤 샘플링을 위한 리스트 (index 저장)
        self.data_map = {} # dict mapping index_item -> (list_index)

    def add_or_update(self, index):
        """데이터를 추가하거나 업데이트하며, 용량 초과로 삭제된 인덱스가 있다면 반환합니다."""
        if index in self.data_map:
            return None
        
        if len(self.items) < self.max_size:
            list_idx = len(self.items)
            self.items.append(index)
            self.data_map[index] = list_idx
            return None
        else:
            # Replace randomly
            replace_list_idx = random.randrange(self.max_size)
            old_index = self.items[replace_list_idx]
            del self.data_map[old_index]
            
            self.items[replace_list_idx] = index
            self.data_map[index] = replace_list_idx
            return old_index

    def remove(self, index):
        """특정 인덱스를 O(1) 시간에 삭제합니다."""
        if index not in self.data_map:
            return
        list_idx = self.data_map[index]
        last_index = self.items[-1]
        
        # swap with last
        self.items[list_idx] = last_index
        self.data_map[last_index] = list_idx
        
        self.items.pop()
        del self.data_map[index]

    def sample(self, k, exclude=None):
        """무작위로 최대 k개를 샘플링하되 제외할 인덱스가 있으면 제외"""
        if not self.items:
            return []
        
        pool = self.items
        if exclude is not None and exclude in self.data_map:
            # exclude가 리스트에 있다면 제외하고 샘플링 (스왑 기법 등을 쓸 수 있지만 단순 복사/필터링 사용)
            pool = [x for x in self.items if x != exclude]
            
        k = min(k, len(pool))
        if k == 0:
            return []
            
        indices = random.sample(pool, k)
        return indices

    def __len__(self):
        return len(self.items)


class RetrievalContextManager:
    _ema_fields = {
        "td_error": ("ema_mean", "ema_var"),
        "value_diff": ("ema_vd_mean", "ema_vd_var"),
        "value": ("ema_value_mean", "ema_value_var"),
    }
    _state_fields = ("hash_proj", "hash_mean", "ema_mean", "ema_var", "ema_vd_mean",
                     "ema_vd_var", "ema_value_mean", "ema_value_var", "hash_memory",
                     "index_to_bucket", "prev_v", "prev_keys", "active_anchors")

    def __init__(self, num_envs, config, latent_dim, device="cuda", statistics_mode="legacy"):
        if statistics_mode not in ("legacy", "batched", "vectorized"):
            raise ValueError(f"Unknown retrieval statistics mode: {statistics_mode!r}")
        self.statistics_mode = statistics_mode
        self.num_envs = num_envs
        self.config = config
        self.device = device
        self.mode = parse_retrieval_mode(config.get("enable", False), allow_experiments=True)
        self.save_warmup = config.get("save_warmup", False)
        self.enabled = (any(RETRIEVAL_EXPERIMENTS[name]["enable"] for name in self.mode)
                        if isinstance(self.mode, list) else self.mode is not False)
        self.enabled = self.enabled or self.save_warmup
        self.threshold = float(config.get("threshold", 1.0))
        self.context_length = int(config.get("context_length", 8))
        self.max_bucket_size = int(config.get("max_bucket_size", 512))
        self.anchor_weight = float(config.get("anchor_weight", 0.5))
        self.anchor_weight_r = None
        relative_weight = config.get("anchor_weight_r")
        if relative_weight is not None:
            try:
                ratio = float(relative_weight)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Retrieval.anchor_weight_r must be null or a finite positive number") from error
            if isinstance(relative_weight, bool) or not math.isfinite(ratio) or ratio <= 0:
                raise ValueError("Retrieval.anchor_weight_r must be null or a finite positive number")
            self.anchor_weight_r = ratio
        self.neighbor_source = config.get("neighbor_source", "retrieval")
        if self.neighbor_source not in ("retrieval", "random"):
            raise ValueError(f"Retrieval.neighbor_source must be retrieval or random; got {self.neighbor_source!r}")
        # Replacement draws must not shift hash candidate sampling or the other
        # training RNG streams. Save this independent stream with checkpoints.
        self._random_neighbor_rng = random.Random(0)
        self._random_neighbor_rng.setstate(random.getstate())
        
        self.trigger_mode = config.get("trigger_mode", "absolute")
        self.value_signal = config.get("value_signal", "value_diff")
        if self.value_signal not in ("value_diff", "value"):
            raise ValueError(f"Retrieval.value_signal must be value_diff or value; got {self.value_signal!r}")
        self.score_combination = config.get("score_combination", "multiply")
        if self.score_combination not in ("multiply", "add"):
            raise ValueError(f"Retrieval.score_combination must be multiply or add; got {self.score_combination!r}")
        # Shared warmup collects the union needed by enabled experiments. Child
        # configurations are boolean and collect only their own value signal.
        value_signals = [self.value_signal]
        if isinstance(self.mode, list):
            value_signals = [RETRIEVAL_EXPERIMENTS[name].get("value_signal", self.value_signal)
                             for name in self.mode if RETRIEVAL_EXPERIMENTS[name]["enable"]]
        if self.save_warmup:
            value_signals = ["value_diff", "value"]
        self._statistics_signals = ("td_error", *dict.fromkeys(value_signals))
        self.anchor_offset = int(config.get("anchor_offset", -2))
        self.z_score_threshold = float(config.get("z_score_threshold", 2.0))
        self.additive_z_score_threshold = float(config.get("additive_z_score_threshold", 4.0))
        self.ema_alpha = float(config.get("ema_alpha", 0.01))
        
        for mean_field, var_field in self._ema_fields.values():
            setattr(self, mean_field, np.zeros(num_envs))
            setattr(self, var_field, np.ones(num_envs))
        
        # Hashing config
        self.hash_bits = int(config.get("hash_bits", 12))
        self.hash_sample_mode = config.get("hash_sample_mode", "probs")
        self.use_pca = config.get("use_pca", True)
        self.max_pca_samples = int(config.get("max_pca_samples", 10000))
        
        proj = torch.randn(latent_dim, self.hash_bits, dtype=torch.float32, device=device)
        self.hash_proj = proj
        self.hash_mean = None
        
        bit_values = 2 ** torch.arange(self.hash_bits, dtype=torch.int64, device=device)
        self.hash_bit_values = bit_values
        
        self.hash_memory = {} # key -> FastHashBucket
        
        # Mapping (pointer, env_idx) -> hash_key (to remove old keys when overwritten)
        self.index_to_bucket = {} 
        
        self.prev_v = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.prev_keys = [-1] * num_envs
        
        self.active_anchors = deque()

    def state_dict(self):
        """Portable retrieval state, including warmup statistics and hash memory."""
        state = {name: (getattr(self, name).detach().cpu().clone()
                       if isinstance(getattr(self, name), torch.Tensor)
                       else copy.deepcopy(getattr(self, name))) for name in self._state_fields}
        state["random_neighbor_rng_state"] = self._random_neighbor_rng.getstate()
        return state

    def load_state_dict(self, state):
        # Keep the receiving branch's scoring configuration, enable flag and device.
        # Older checkpoints have no value statistics; use independent priors.
        for mean_field, var_field in self._ema_fields.values():
            if mean_field not in state:
                setattr(self, mean_field, np.zeros(self.num_envs))
            if var_field not in state:
                setattr(self, var_field, np.ones(self.num_envs))
        for name in self._state_fields:
            if name in state:
                value = state[name]
                setattr(self, name, value.to(self.device) if isinstance(value, torch.Tensor)
                        else copy.deepcopy(value))
        if "random_neighbor_rng_state" in state:
            self._random_neighbor_rng.setstate(state["random_neighbor_rng_state"])

    @property
    def statistics_signals(self):
        return self._statistics_signals

    def get_ema_stats(self, signal):
        """Return the named signal's mean and variance without mixing states."""
        mean_field, var_field = self._ema_fields[signal]
        return getattr(self, mean_field), getattr(self, var_field)

    def _z_score(self, values, signal, env_indices):
        mean, var = self.get_ema_stats(signal)
        mean_t = torch.from_numpy(mean).to(values.device, dtype=torch.float32)
        var_t = torch.from_numpy(var).to(values.device, dtype=torch.float32)
        batch_mean = mean_t[env_indices].unsqueeze(1).expand_as(values)
        batch_var = var_t[env_indices].unsqueeze(1).expand_as(values)
        return (values - batch_mean) / (torch.sqrt(batch_var) + 1e-8)

    def _update_ema_stats(self, signals, valid_mask, env_indices):
        """Update configured statistics from valid samples, separately per env."""
        if self.statistics_mode == "batched":
            return self._update_ema_stats_batched(signals, valid_mask, env_indices)
        if self.statistics_mode == "vectorized":
            return self._update_ema_stats_vectorized(signals, valid_mask, env_indices)
        if not valid_mask.any():
            return
        valid_envs = env_indices.unsqueeze(1).expand_as(valid_mask)[valid_mask]
        valid_signals = [
            (signals[name][valid_mask], *self.get_ema_stats(name))
            for name in self._statistics_signals
        ]
        for i in range(self.num_envs):
            env_mask = valid_envs == i
            if not env_mask.any():
                continue
            for values, mean, var in valid_signals:
                env_values = values[env_mask]
                env_mean = env_values.mean().item()
                env_var = env_values.var(unbiased=False).item() if env_values.numel() > 1 else 0.0
                diff = env_mean - mean[i]
                mean[i] += self.ema_alpha * diff
                var[i] = (1 - self.ema_alpha) * var[i] + self.ema_alpha * (env_var + diff**2)

    def _update_ema_stats_batched(self, signals, valid_mask, env_indices):
        """Same reductions and CPU float64 recurrence; one scalar transfer."""
        if not valid_mask.any():
            return
        valid_envs = env_indices.unsqueeze(1).expand_as(valid_mask)[valid_mask]
        valid_signals = [
            (signals[name][valid_mask], *self.get_ema_stats(name))
            for name in self._statistics_signals
        ]
        scalars, destinations = [], []
        for i in range(self.num_envs):
            env_mask = valid_envs == i
            if not env_mask.any():
                continue
            for values, mean, var in valid_signals:
                env_values = values[env_mask]
                scalars.extend((env_values.mean(), env_values.var(unbiased=False)
                                if env_values.numel() > 1 else env_values.new_zeros(())))
                destinations.append((i, mean, var))
        if not scalars:
            return
        host = torch.stack(scalars).detach().cpu().tolist()
        for j, (i, mean, var) in enumerate(destinations):
            env_mean, env_var = host[2*j:2*j+2]
            diff = env_mean - mean[i]
            mean[i] += self.ema_alpha * diff
            var[i] = (1 - self.ema_alpha) * var[i] + self.ema_alpha * (env_var + diff**2)

    def _update_ema_stats_vectorized(self, signals, valid_mask, env_indices):
        """Fixed-shape GPU reductions; preserve population variance and CPU EMA.

        Masked-out values (including NaNs) never contribute. Unlike the legacy
        path, this needs no CUDA scalar predicates or dynamic boolean gathers.
        Reduction order differs, so equivalence is mathematical, not bitwise.
        """
        values = torch.stack([signals[name] for name in self._statistics_signals])
        # Accumulate low-precision input in FP32, and retain FP64 input precision.
        if values.dtype != torch.float64:
            values = values.float()
        envs = torch.arange(self.num_envs, device=env_indices.device)
        mask = valid_mask.unsqueeze(0) & (env_indices[None, :, None] == envs[:, None, None])
        counts = mask.sum(dim=(-2, -1))
        denom = counts.clamp_min(1).to(values.dtype).unsqueeze(1)
        selected = torch.where(mask[:, None], values[None], 0)
        means = selected.sum(dim=(-2, -1)) / denom
        centered = torch.where(mask[:, None], values[None] - means[:, :, None, None], 0)
        variances = centered.square().sum(dim=(-2, -1)) / denom
        variances = torch.where(counts[:, None] > 1, variances, 0)
        host = torch.cat((counts[:, None].double(), means.double(), variances.double()), dim=1)
        host = host.detach().cpu().tolist()
        n = len(self._statistics_signals)
        for i, row in enumerate(host):
            if row[0] == 0:
                continue
            for j, name in enumerate(self._statistics_signals):
                mean, var = self.get_ema_stats(name)
                env_mean, env_var = row[1+j], row[1+n+j]
                diff = env_mean - mean[i]
                mean[i] += self.ema_alpha * diff
                var[i] = (1 - self.ema_alpha) * var[i] + self.ema_alpha * (env_var + diff**2)

    def _valid_context(self, replay_buffer, pointer, env_idx):
        if hasattr(replay_buffer, "is_valid_context"):
            return replay_buffer.is_valid_context(pointer, env_idx, self.context_length)
        capacity = replay_buffer.max_length // replay_buffer.num_envs
        length = min(replay_buffer.length, capacity)
        if not 0 <= env_idx < replay_buffer.num_envs or not 0 <= pointer < capacity:
            return False
        oldest = (replay_buffer.last_pointer + 1) % capacity if length == capacity else 0
        position = (pointer - oldest) % capacity
        if position >= length or position + 1 < self.context_length:
            return False
        return all(replay_buffer.termination_buffer[(pointer - step) % capacity, env_idx] <= 0.5
                   for step in range(1, self.context_length))

    def _hash_keys(self, latent):
        if latent.numel() == 0:
            return []
            
        latent_f = latent.float()
        if self.hash_mean is not None:
            latent_f = latent_f - self.hash_mean.float()
            
        scores = latent_f @ self.hash_proj.float()
        bits = scores > 0
        keys = (bits.to(torch.int64) * self.hash_bit_values).sum(dim=-1)
        return keys.detach().cpu().tolist()

    def add_batch_transitions(self, v_t, reward, termination, gamma, base_indexes, base_envs, max_buf_len, skip_len=8, is_warmup=False, transition_mask=None):
        if not self.enabled or skip_len < 1 or v_t.shape[1] <= skip_len + 1:
            return 0
            
        v_t_eval = v_t[:, skip_len:]
        reward_eval = reward[:, skip_len:].squeeze(-1) if reward.dim() == 3 else reward[:, skip_len:]
        term_eval = termination[:, skip_len:].squeeze(-1) if termination.dim() == 3 else termination[:, skip_len:]
        
        v_curr = v_t_eval[:, :-1]
        v_next = v_t_eval[:, 1:]
        r_curr = reward_eval[:, :-1]
        d_curr = term_eval[:, :-1]
        
        target_v = r_curr + gamma * v_next * (1.0 - d_curr)
        delta_v_raw = target_v - v_curr
        
        valid_mask_1d = torch.from_numpy(base_envs != -1).to(delta_v_raw.device)
        if not valid_mask_1d.any():
            return 0
            
        valid_mask_2d = valid_mask_1d.unsqueeze(1).expand_as(delta_v_raw)
        # Optional [B, T-1] mask excludes reset edges and unavailable contexts.
        if transition_mask is not None:
            valid_mask_2d = valid_mask_2d & transition_mask[:, skip_len:].to(
                device=delta_v_raw.device, dtype=torch.bool)
        env_indices_full = torch.from_numpy(base_envs).to(delta_v_raw.device)
        
        abs_delta_v = torch.abs(delta_v_raw)
        
        metric = torch.full_like(delta_v_raw, float('-inf'))
        
        if getattr(self, "trigger_mode", "absolute") == "z_score" or (self.save_warmup and is_warmup):
            signals = {
                "td_error": abs_delta_v,
                "value_diff": v_curr - v_t[:, skip_len-1:-2],
                "value": v_curr,
            }
            if is_warmup:
                self._update_ema_stats(signals, valid_mask_2d, env_indices_full)
                return 0
            # Score against the previous EMA, then incorporate this batch.
            z_td_full = self._z_score(signals["td_error"], "td_error", env_indices_full)
            z_value_full = self._z_score(signals[self.value_signal], self.value_signal, env_indices_full)
            td_score = torch.relu(z_td_full)
            value_score = torch.nn.functional.softplus(z_value_full)
            if self.score_combination == "add":
                combined_metric = td_score + value_score
                threshold = self.additive_z_score_threshold
            else:
                combined_metric = td_score * value_score
                threshold = self.z_score_threshold
            metric[valid_mask_2d] = combined_metric[valid_mask_2d]
            self._update_ema_stats(signals, valid_mask_2d, env_indices_full)
        else:
            metric[valid_mask_2d] = abs_delta_v[valid_mask_2d]
            threshold = self.threshold

        max_val, max_idx = metric.max(dim=1)
        triggered_b = max_val >= threshold
        
        def process_max_triggers(triggered_b, max_idx):
            num_trig = 0
            if triggered_b.any():
                b_indices = triggered_b.nonzero(as_tuple=True)[0]
                t_indices = max_idx[b_indices]
                
                for i in range(len(b_indices)):
                    orig_b_idx = b_indices[i].item()
                    t_idx = t_indices[i].item()
                    
                    env_idx = base_envs[orig_b_idx]
                    base_ptr = base_indexes[orig_b_idx]
                    
                    # The score belongs to v_curr at skip_len + t_idx.
                    # Apply the signed offset directly from that replay position.
                    anchor_ptr = int(base_ptr + skip_len + t_idx + self.anchor_offset) % max_buf_len
                    
                    anchor_key = self.index_to_bucket.get((anchor_ptr, env_idx), -1)
                    if anchor_key != -1:
                        anchor = (anchor_ptr, env_idx)
                        self.active_anchors.append((anchor, anchor_key))
                        num_trig += 1
            return num_trig
            
        if is_warmup:
            return 0
            
        num_triggered = process_max_triggers(triggered_b, max_idx)
        
        return num_triggered

    def add_transition(self, pointer, env_idx, latent_b):
        """
        pointer: the buffer pointer at which current_obs was just appended.
        latent_b: (num_envs, latent_dim)
        """
        if not self.enabled:
            return 0
            
        keys = self._hash_keys(latent_b)
        
        for i in range(self.num_envs):
            if env_idx != -1 and i != env_idx:
                continue # for individual updates if needed, though usually batch
                
            self.prev_keys[i] = keys[i]
            
            # Add current latent to hash bucket
            self._insert_into_bucket(pointer, i, keys[i])
            
        return 0

    def _insert_into_bucket(self, pointer, env_idx, key):
        idx_tuple = (pointer, env_idx)
        old_key = self.index_to_bucket.get(idx_tuple, -1)
        
        if old_key != -1 and old_key != key:
            old_queue = self.hash_memory.get(old_key)
            if old_queue is not None:
                old_queue.remove(idx_tuple)
                
        queue = self.hash_memory.get(key)
        if queue is None:
            queue = FastHashBucket(max_size=self.max_bucket_size)
            self.hash_memory[key] = queue
            
        evicted_idx = queue.add_or_update(idx_tuple)
        if evicted_idx is not None:
            if evicted_idx in self.index_to_bucket:
                del self.index_to_bucket[evicted_idx]
                
        self.index_to_bucket[idx_tuple] = key

    def _sample_random_neighbors(self, replay_buffer, count, exclude):
        """Uniform valid replay endpoints, excluding the anchor, without replacement.

        A lazy Fisher-Yates shuffle avoids scanning/copying the whole replay on
        each group. Rejection applies the same context validity rules as retrieval;
        random contexts may naturally belong to the anchor's hash bucket.
        """
        if count == 0:
            return []
        capacity = replay_buffer.max_length // replay_buffer.num_envs
        length = min(replay_buffer.length, capacity)
        remaining = length * replay_buffer.num_envs
        swaps = {}
        neighbors = []
        while remaining and len(neighbors) < count:
            slot = self._random_neighbor_rng.randrange(remaining)
            candidate = swaps.get(slot, slot)
            remaining -= 1
            swaps[slot] = swaps.get(remaining, remaining)
            pointer = divmod(candidate, replay_buffer.num_envs)
            if pointer != exclude and self._valid_context(replay_buffer, *pointer):
                neighbors.append(pointer)
        if len(neighbors) != count:
            raise RuntimeError("Not enough valid random contexts to replace verified neighbors")
        return neighbors

    def retrieve_contexts(self, replay_buffer, world_model, max_anchors, multiplier=5, target=5, max_contexts=256, return_indices=False):
        """
        Pops up to `max_anchors` from `active_anchors` and retrieves up to max contexts in total.
        Implements lazy recomputation using single frame encoding.
        """
        def result(*values, indices=None):
            return (*values, indices or []) if return_indices else values
        if not self.enabled or len(self.active_anchors) == 0 or max_contexts <= 0:
            return result(None, None, 0, 0.0, [], 0)
        if hasattr(replay_buffer, "retrieval_view"):
            replay_buffer = replay_buffer.retrieval_view()
        gather_frames = getattr(replay_buffer, "gather_retrieval_frames", None)
        gather_contexts = getattr(replay_buffer, "gather_retrieval_contexts", None)
            
        popped_anchors = []
        for _ in range(min(max_anchors, len(self.active_anchors))):
            popped_anchors.append(self.active_anchors.popleft())
            
        retrieved_obs_list = []
        retrieved_action_list = []
        
        max_buf_len = replay_buffer.max_length // replay_buffer.num_envs
        
        anchor_groups = []
        
        total_hit_rate = 0.0
        num_hit_rate_samples = 0
        
        for anchor_tuple, anchor_key in popped_anchors:
            # anchor_tuple is (anchor_ptr, env_idx)
            anchor_ptr, anchor_env_idx = anchor_tuple
            if not self._valid_context(replay_buffer, anchor_ptr, anchor_env_idx):
                continue
            queue = self.hash_memory.get(anchor_key)
            if not queue:
                continue
                
            sampled_indices = queue.sample(k=multiplier*(target-1), exclude=(anchor_ptr, anchor_env_idx))
                
            obs_list = []
            valid_sampled = []
            for (p, env_idx) in sampled_indices:
                curr_p = p % max_buf_len
                if not self._valid_context(replay_buffer, p, env_idx):
                    continue
                if gather_frames is None:
                    obs_list.append(replay_buffer.obs_buffer[curr_p, env_idx:env_idx+1])
                valid_sampled.append((p, env_idx))
                
            if valid_sampled:
                frames = gather_frames(valid_sampled) if gather_frames is not None else None
                if replay_buffer.store_on_gpu:
                    if frames is None:
                        frames = torch.cat(obs_list, dim=0)
                    obs_tensor = frames.float() / 255.0
                else:
                    import numpy as np
                    if frames is None:
                        frames = np.concatenate(obs_list, axis=0)
                    obs_tensor = torch.from_numpy(frames).to(self.device).float() / 255.0
                    
                from einops import rearrange
                obs_tensor = rearrange(obs_tensor, "N H W C -> N 1 C H W")
                
                with torch.no_grad():
                    encoded = world_model.encode_obs(obs_tensor, sample_mode=self.hash_sample_mode) # [N, 1, latent_dim]
                    encoded = encoded.squeeze(1) # [N, latent_dim]
                    
                current_keys = self._hash_keys(encoded)
                
                if len(current_keys) > 0:
                    hit_rate = (torch.tensor(current_keys) == anchor_key).float().mean().item()
                    total_hit_rate += hit_rate
                    num_hit_rate_samples += 1
            else:
                current_keys = []
            
            # Filter matches
            matched_indices = [(anchor_ptr, anchor_env_idx)] # 앵커를 항상 첫 번째 타겟으로 추가
            for i_c, k_c in enumerate(current_keys):
                if k_c == anchor_key:
                    matched_indices.append(valid_sampled[i_c])
                    if len(matched_indices) >= target:
                        break
                        
            if matched_indices:
                anchor_groups.append(matched_indices)
            
        candidates_before_max = sum(len(g) for g in anchor_groups)
        
        final_anchor_groups = []
        total_candidates = 0
        for group in anchor_groups:
            if total_candidates >= max_contexts:
                break
            space = max_contexts - total_candidates
            to_add = group[:space]
            total_candidates += len(to_add)
            final_anchor_groups.append(to_add)
            
        retrieved_obs_list = []
        retrieved_action_list = []
        retrieved_weights = []
        retrieved_indices = []
        valid_anchors_count = 0
        
        for group in final_anchor_groups:
            valid_group_obs = []
            valid_group_action = []
            valid_is_anchor = []

            # Establish the final verified group before changing its contents.
            # Its anchor, size, member roles and weights remain unchanged.
            valid_members = [(idx == 0, pointer) for idx, pointer in enumerate(group)
                             if self._valid_context(replay_buffer, *pointer)]
            if self.neighbor_source == "random":
                replacements = iter(self._sample_random_neighbors(
                    replay_buffer, sum(not is_anchor for is_anchor, _ in valid_members),
                    exclude=group[0]))
                valid_members = [(is_anchor, pointer if is_anchor else next(replacements))
                                 for is_anchor, pointer in valid_members]
            
            for is_anchor, (p, env_idx) in valid_members:
                valid_is_anchor.append(is_anchor)
                retrieved_indices.append((p, env_idx))
                if gather_contexts is not None:
                    continue
                obs_chunk = []
                action_chunk = []
                for step in range(self.context_length - 1, -1, -1):
                    curr_p = (p - step) % max_buf_len
                    obs_chunk.append(replay_buffer.obs_buffer[curr_p, env_idx:env_idx+1])
                    action_chunk.append(replay_buffer.action_buffer[curr_p, env_idx:env_idx+1])

                if replay_buffer.store_on_gpu:
                    obs_tensor = torch.stack(obs_chunk, dim=0)
                    action_tensor = torch.stack(action_chunk, dim=0)
                else:
                    import numpy as np
                    obs_tensor = np.stack(obs_chunk, axis=0)
                    action_tensor = np.stack(action_chunk, axis=0)

                valid_group_obs.append(obs_tensor)
                valid_group_action.append(action_tensor)
                
            if valid_is_anchor:
                valid_anchors_count += 1
                
                L = len(valid_is_anchor)
                has_anchor = any(valid_is_anchor)
                
                if L == 1:
                    weights = [1.0]
                elif has_anchor:
                    num_targets = L - 1
                    if self.anchor_weight_r is None:
                        anchor_weight = self.anchor_weight
                        neighbor_weight = (1.0 - self.anchor_weight) / num_targets
                    else:
                        # Use the final verified/capped group, not the requested target.
                        denominator = num_targets + self.anchor_weight_r
                        anchor_weight = self.anchor_weight_r / denominator
                        neighbor_weight = 1.0 / denominator
                    weights = [anchor_weight if is_a else neighbor_weight for is_a in valid_is_anchor]
                else:
                    weights = [1.0 / L] * L
                    
                for w in weights:
                    retrieved_weights.append(w)
                
                retrieved_obs_list.extend(valid_group_obs)
                retrieved_action_list.extend(valid_group_action)
                
        avg_hit_rate = total_hit_rate / max(1, num_hit_rate_samples)
        if len(retrieved_indices) == 0:
            return result(None, None, candidates_before_max, avg_hit_rate, [], 0)

        if gather_contexts is not None:
            context_obs, context_action = gather_contexts(retrieved_indices, self.context_length)
        elif replay_buffer.store_on_gpu:
            context_obs = torch.cat(retrieved_obs_list, dim=1)
            context_action = torch.cat(retrieved_action_list, dim=1)
        else:
            context_obs = np.concatenate(retrieved_obs_list, axis=1)
            context_action = np.concatenate(retrieved_action_list, axis=1)
            
        if replay_buffer.store_on_gpu:
            ret_obs = context_obs.float() / 255.0
            from einops import rearrange
            ret_obs = rearrange(ret_obs, "T B H W C -> B T C H W")
            ret_action = context_action.transpose(0, 1) # [B, T]
        else:
            import numpy as np
            ret_obs = torch.from_numpy(context_obs).to(self.device).float() / 255.0
            from einops import rearrange
            ret_obs = rearrange(ret_obs, "T B H W C -> B T C H W")
            ret_action = torch.from_numpy(context_action).to(self.device).transpose(0, 1) # [B, T, ...]
            
        return result(ret_obs, ret_action, candidates_before_max, avg_hit_rate, retrieved_weights, valid_anchors_count, indices=retrieved_indices)

    @torch.no_grad()
    def rebuild_all_hash_buckets(self, replay_buffer, world_model, chunk_size=8192):
        """
        Clears all hash buckets and re-hashes all valid transitions in the replay buffer 
        using the latest world model to fix representation drift (Global Rebuild).
        """
        if not self.enabled:
            return
        if hasattr(replay_buffer, "retrieval_view"):
            replay_buffer = replay_buffer.retrieval_view()
        if hasattr(replay_buffer, "rebuild_hash_buckets"):
            replay_buffer.rebuild_hash_buckets(self, world_model, chunk_size)
            return
        self.hash_memory.clear()
        self.index_to_bucket.clear()
        
        if replay_buffer.length < self.context_length:
            return
            
        max_buf_len = replay_buffer.max_length // replay_buffer.num_envs
        valid_len = min(replay_buffer.length, max_buf_len)
        
        all_encoded_latents = []
        all_valid_indices = []
        
        # 1. Collect all valid latents in chunks. 
        # Although GPU VRAM is large, we must avoid massive vectorized slicing (e.g. buffer[:valid_len]) 
        # on the Replay Buffer due to shared-memory/memmap silent read failures (yielding all zeros).
        for start_idx in range(0, valid_len, chunk_size):
            end_idx = min(start_idx + chunk_size, valid_len)
            
            obs_list = []
            valid_indices = []
            
            # Safe iterative read to prevent memory-mapped array zero-read issues
            for p in range(start_idx, end_idx):
                for env_idx in range(replay_buffer.num_envs):
                    obs_list.append(replay_buffer.obs_buffer[p, env_idx:env_idx+1])
                    valid_indices.append((p, env_idx))
                    
            if len(obs_list) == 0:
                continue
                
            if replay_buffer.store_on_gpu:
                obs_tensor = torch.cat(obs_list, dim=0).float() / 255.0
            else:
                import numpy as np
                obs_arr = np.concatenate(obs_list, axis=0)
                obs_tensor = torch.from_numpy(obs_arr).to(self.device).float() / 255.0
                
            from einops import rearrange
            obs_tensor = rearrange(obs_tensor, "N H W C -> N 1 C H W")
            
            encoded = world_model.encode_obs(obs_tensor, sample_mode=self.hash_sample_mode)
            encoded = encoded.squeeze(1)
            
            all_encoded_latents.append(encoded)
            all_valid_indices.extend(valid_indices)
        
        if len(all_encoded_latents) == 0:
            return
            
        full_latents = torch.cat(all_encoded_latents, dim=0)
        
        # 2. PCA Projection Update
        if self.use_pca and full_latents.shape[0] > self.hash_bits:
            pca_input = full_latents
            if pca_input.shape[0] > self.max_pca_samples:
                indices = torch.randperm(pca_input.shape[0], device=pca_input.device)[:self.max_pca_samples]
                pca_input = pca_input[indices]
                
            mean_latent = pca_input.mean(dim=0, keepdim=True)
            centered_input = pca_input - mean_latent
            
            # Perform PCA (q=hash_bits)
            U, S, V = torch.pca_lowrank(centered_input, q=self.hash_bits, center=False)
            
            if not torch.isnan(V).any():
                self.hash_proj = V
                self.hash_mean = mean_latent
                
        # 3. Bulk Hashing and Insertion
        keys = self._hash_keys(full_latents)
        
        for (p, env_idx), key in zip(all_valid_indices, keys):
            self._insert_into_bucket(p, env_idx, key)
                
        print(f"[Retrieval] Global Rebuild completed. Re-hashed {valid_len * replay_buffer.num_envs} frames (PCA: {self.use_pca}).")
