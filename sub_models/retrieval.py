import torch
import random
import numpy as np
from collections import deque

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
    def __init__(self, num_envs, config, latent_dim, device="cuda"):
        self.num_envs = num_envs
        self.device = device
        self.enabled = bool(config.get("enable", True))
        self.threshold = float(config.get("threshold", 5.0))
        self.max_retrievals = int(config.get("max_retrievals", 4))
        self.context_length = int(config.get("context_length", 8))
        self.max_bucket_size = int(config.get("max_bucket_size", 512))
        
        self.trigger_mode = config.get("trigger_mode", "absolute")
        self.anchor_offset = int(config.get("anchor_offset", -2))
        self.z_score_threshold = float(config.get("z_score_threshold", 2.0))
        self.ema_alpha = float(config.get("ema_alpha", 0.01))
        
        self.ema_mean = np.zeros(num_envs)
        self.ema_var = np.ones(num_envs)
        
        # Hashing config
        self.hash_bits = 12
        proj = torch.randn(latent_dim, self.hash_bits, dtype=torch.float32, device=device)
        self.hash_proj = proj
        bit_values = 2 ** torch.arange(self.hash_bits, dtype=torch.int64, device=device)
        self.hash_bit_values = bit_values
        
        self.hash_memory = {} # key -> FastHashBucket
        
        # Mapping (pointer, env_idx) -> hash_key (to remove old keys when overwritten)
        self.index_to_bucket = {} 
        
        self.prev_v = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.prev_keys = [-1] * num_envs
        
        self.active_anchors = deque()

    def _hash_keys(self, latent):
        if latent.numel() == 0:
            return []
        scores = latent.float() @ self.hash_proj.float()
        bits = scores > 0
        keys = (bits.to(torch.int64) * self.hash_bit_values).sum(dim=-1)
        return keys.detach().cpu().tolist()

    def add_transition(self, pointer, env_idx, latent_b, value_b, is_first_step_b):
        """
        pointer: the buffer pointer at which current_obs was just appended.
        latent_b: (num_envs, latent_dim)
        value_b: (num_envs,)
        is_first_step_b: (num_envs,) bool array (True if the environment just reset)
        """
        if not self.enabled:
            return 0
            
        keys = self._hash_keys(latent_b)
        num_triggered = 0
        
        for i in range(self.num_envs):
            if env_idx != -1 and i != env_idx:
                continue # for individual updates if needed, though usually batch
                
            v_t = value_b[i].item()
            is_first = is_first_step_b[i]
            
            if not is_first:
                # Compute delta_v against prev_v
                delta_v = abs(v_t - self.prev_v[i].item())
                triggered = False
                
                if getattr(self, "trigger_mode", "absolute") == "z_score":
                    diff = delta_v - self.ema_mean[i]
                    self.ema_mean[i] += self.ema_alpha * diff
                    self.ema_var[i] = (1 - self.ema_alpha) * (self.ema_var[i] + self.ema_alpha * diff ** 2)
                    
                    import math
                    z_score = abs(delta_v - self.ema_mean[i]) / (math.sqrt(self.ema_var[i]) + 1e-8)
                    if z_score >= self.z_score_threshold:
                        triggered = True
                else:
                    if delta_v >= self.threshold:
                        triggered = True
                        
                if triggered:
                    anchor_ptr = pointer + self.anchor_offset
                    anchor_key = self.index_to_bucket.get((anchor_ptr, i), -1)
                    if anchor_key != -1:
                        anchor = (anchor_ptr, i)
                        self.active_anchors.append((anchor, anchor_key))
                        num_triggered += 1
                    
            self.prev_v[i] = v_t
            self.prev_keys[i] = keys[i]
            
            # Add current latent to hash bucket
            self._insert_into_bucket(pointer, i, keys[i])
            
        return num_triggered

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

    def retrieve_contexts(self, replay_buffer, world_model, max_anchors, x=5, y=5, z=256):
        """
        Pops up to `max_anchors` from `active_anchors` and retrieves up to z contexts in total.
        Implements lazy recomputation using single frame encoding.
        """
        if not self.enabled or len(self.active_anchors) == 0:
            return None, None, 0
            
        popped_anchors = []
        for _ in range(min(max_anchors, len(self.active_anchors))):
            popped_anchors.append(self.active_anchors.popleft())
            
        retrieved_obs_list = []
        retrieved_action_list = []
        
        max_buf_len = replay_buffer.max_length // replay_buffer.num_envs
        final_chosen_indices = []
        
        for anchor_tuple, anchor_key in popped_anchors:
            queue = self.hash_memory.get(anchor_key)
            if not queue:
                continue
                
            sampled_indices = queue.sample(k=x*(y-1), exclude=anchor_tuple)
            if not sampled_indices:
                continue
                
            obs_list = []
            valid_sampled = []
            for (p, env_idx) in sampled_indices:
                curr_p = p % max_buf_len
                if not replay_buffer.store_on_gpu and p < 0 and replay_buffer.length < max_buf_len:
                    continue
                if replay_buffer.length < self.context_length:
                    continue
                obs_list.append(replay_buffer.obs_buffer[curr_p, env_idx:env_idx+1])
                valid_sampled.append((p, env_idx))
                
            if not obs_list:
                continue
                
            if replay_buffer.store_on_gpu:
                obs_tensor = torch.cat(obs_list, dim=0).float() / 255.0
            else:
                import numpy as np
                obs_arr = np.concatenate(obs_list, axis=0)
                obs_tensor = torch.from_numpy(obs_arr).float().cuda() / 255.0
                
            from einops import rearrange
            obs_tensor = rearrange(obs_tensor, "N H W C -> N 1 C H W")
            
            with torch.no_grad():
                encoded = world_model.encode_obs(obs_tensor) # [N, 1, latent_dim]
                encoded = encoded.squeeze(1) # [N, latent_dim]
                
            current_keys = self._hash_keys(encoded)
            
            # Filter matches
            matched_indices = []
            for i_c, k_c in enumerate(current_keys):
                if k_c == anchor_key:
                    matched_indices.append(valid_sampled[i_c])
                    if len(matched_indices) >= (y - 1):
                        break
                        
            final_chosen_indices.extend(matched_indices)
            
        candidates_before_z = len(final_chosen_indices)
        if len(final_chosen_indices) > z:
            final_chosen_indices = final_chosen_indices[:z]
            
        retrieved_obs_list = []
        retrieved_action_list = []
        
        for (p, env_idx) in final_chosen_indices:
            valid = True
            obs_chunk = []
            action_chunk = []
            for step in range(self.context_length - 1, -1, -1):
                curr_p = (p - step) % max_buf_len
                term = replay_buffer.termination_buffer[curr_p, env_idx]
                if step > 0 and term > 0.5:
                    valid = False
                    break
                    
            if valid:
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
                    
                retrieved_obs_list.append(obs_tensor)
                retrieved_action_list.append(action_tensor)
                
        if len(retrieved_obs_list) == 0:
            return None, None, candidates_before_z
            
        if replay_buffer.store_on_gpu:
            ret_obs = torch.cat(retrieved_obs_list, dim=1).float() / 255.0
            from einops import rearrange
            ret_obs = rearrange(ret_obs, "T B H W C -> B T C H W")
            ret_action = torch.cat(retrieved_action_list, dim=1).transpose(0, 1) # [B, T]
        else:
            import numpy as np
            ret_obs = np.concatenate(retrieved_obs_list, axis=1)
            ret_obs = torch.from_numpy(ret_obs).float().cuda() / 255.0
            from einops import rearrange
            ret_obs = rearrange(ret_obs, "T B H W C -> B T C H W")
            ret_action = np.concatenate(retrieved_action_list, axis=1)
            ret_action = torch.from_numpy(ret_action).cuda().transpose(0, 1) # [B, T]
            
        return ret_obs, ret_action, candidates_before_z
