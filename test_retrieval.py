import torch
import numpy as np

# Mock classes for testing
class DummyReplayBuffer:
    def __init__(self, num_envs, max_length):
        self.num_envs = num_envs
        self.max_length = max_length
        self.length = max_length // num_envs
        self.store_on_gpu = False
        
        self.obs_buffer = np.zeros((self.length, num_envs, 3, 64, 64), dtype=np.uint8)
        self.action_buffer = np.zeros((self.length, num_envs), dtype=np.float32)
        self.reward_buffer = np.zeros((self.length, num_envs), dtype=np.float32)
        self.termination_buffer = np.zeros((self.length, num_envs), dtype=np.float32)

from sub_models.retrieval import RetrievalContextManager

class DummyWorldModel:
    def encode_obs(self, obs):
        # mock encode_obs: expects [N, 1, C, H, W]
        N = obs.shape[0]
        return torch.ones(N, 1, 16) # return dummy latent of shape [N, 1, latent_dim]

def test_retrieval():
    num_envs = 2
    latent_dim = 16
    config = {
        "enable": True,
        "threshold": 1.0,
        "max_retrievals": 2,
        "context_length": 4,
        "max_bucket_size": 10
    }
    manager = RetrievalContextManager(num_envs=num_envs, config=config, latent_dim=latent_dim, device="cpu")
    
    buf = DummyReplayBuffer(num_envs=num_envs, max_length=100) # length = 50
    world_model = DummyWorldModel()
    
    # Simulate some transitions
    is_first_step = np.array([False, False])
    
    for step in range(10):
        ptr = step
        # Create dummy latent
        latent = torch.ones(num_envs, latent_dim) * step
        # Create dummy value. At step 5, make a big jump for env 0
        value = torch.zeros(num_envs)
        if step == 5:
            value[0] = 5.0  # Delta = 5.0, should trigger
        
        manager.add_transition(pointer=ptr, env_idx=-1, latent_b=latent, value_b=value, is_first_step_b=is_first_step)
        
    print("Active anchors after loop:", manager.active_anchors)
    assert len(manager.active_anchors) == 1
    anchor_ptr, anchor_env = manager.active_anchors[0][0]
    assert anchor_ptr == 4 # since it triggered at step 5, anchor is 4
    assert anchor_env == 0
    
    # Retrieve contexts with lazy rebuild
    obs, action, c_z = manager.retrieve_contexts(buf, world_model, max_anchors=1, x=2, y=2, z=5)
    
    print("Retrieval Result OBS shape:", obs.shape if obs is not None else None)
    
    if obs is not None:
        assert obs.shape[0] > 0
        assert obs.shape[1] == 4 # context_length
    
    print("Test passed!")

if __name__ == "__main__":
    test_retrieval()
