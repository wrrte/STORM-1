import torch
import os
import numpy as np
import random
from tensorboardX import SummaryWriter
from einops import repeat
from contextlib import contextmanager
import time
import yacs
from yacs.config import CfgNode as CN
import wandb


def seed_np_torch(seed=20010105):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger():
    def __init__(self, path, config=None) -> None:
        self.writer = SummaryWriter(logdir=path, flush_secs=1)
        self.tag_step = {}
        self.wandb_buffer = {}

        # Initialize wandb
        try:
            api_key_path = os.path.join(os.path.dirname(__file__), '.wandb_api_key')
            if os.path.exists(api_key_path):
                with open(api_key_path, 'r') as f:
                    wandb.login(key=f.read().strip())
            
            base_name = path.split('/')[-1] if '/' in path else path
            pure_env_name = base_name.split('-')[0]
            run_id = wandb.util.generate_id()
            run_name = f"{pure_env_name}_{run_id}"
            
            wandb.init(project="STORM", name=run_name, id=run_id, config=config)
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")

    def log(self, tag, value):
        if tag not in self.tag_step:
            self.tag_step[tag] = 0
        else:
            self.tag_step[tag] += 1
        
        step = self.tag_step[tag]
        
        if "video" in tag:
            self.writer.add_video(tag, value, step, fps=15)
            
            # handle 5D NTCHW by taking the first item for wandb and local saving
            vid_val = value[0] if len(value.shape) == 5 else value
            
            try:
                if "openloop_video" in tag:
                    self.wandb_buffer[tag] = wandb.Video(vid_val, fps=15, format='gif')
                else:
                    self.wandb_buffer[tag] = wandb.Video(vid_val, fps=15, format='mp4')
            except Exception:
                pass
            
            if "openloop_video" in tag:
                try:
                    import imageio
                    vid = np.transpose(vid_val, (0, 2, 3, 1))
                    if vid.dtype != np.uint8:
                        vid = (255 * np.clip(vid, 0, 1)).astype(np.uint8)
                    
                    scale = max(1, int(np.round(512 / vid.shape[1])))
                    if scale > 1:
                        vid = np.repeat(np.repeat(vid, scale, axis=1), scale, axis=2)
                        
                    safe_name = tag.replace('/', '_')
                    logdir = getattr(self.writer, 'logdir', '.')
                    filename = os.path.join(logdir, f"{step}_{safe_name}.mp4")
                    os.makedirs(logdir, exist_ok=True)
                    imageio.mimsave(filename, vid, fps=15, macro_block_size=1, quality=10, pixelformat='yuv444p')
                except Exception as e:
                    print(f"Failed to save video locally: {e}")
        elif "images" in tag:
            self.writer.add_images(tag, value, step)
            try:
                images = [wandb.Image(img) for img in value]
                self.wandb_buffer[tag] = images
            except Exception:
                pass
        elif "hist" in tag:
            self.writer.add_histogram(tag, value, step)
            try:
                self.wandb_buffer[tag] = wandb.Histogram(value)
            except Exception:
                pass
        else:
            self.writer.add_scalar(tag, value, step)
            try:
                self.wandb_buffer[tag] = value
            except Exception:
                pass

    def flush_wandb(self):
        """Flush all buffered wandb metrics in a single wandb.log() call."""
        if self.wandb_buffer:
            try:
                wandb.log(self.wandb_buffer)
            except Exception:
                pass
            self.wandb_buffer = {}


class EMAScalar():
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar


def load_config(config_path):
    conf = CN()
    # Task need to be RandomSample/TrainVQVAE/TrainWorldModel
    conf.Task = ""

    conf.BasicSettings = CN()
    conf.BasicSettings.Seed = 0
    conf.BasicSettings.ImageSize = 0
    conf.BasicSettings.ReplayBufferOnGPU = False

    # Under this setting, input 128*128 -> latent 16*16*64
    conf.Models = CN()

    conf.Models.WorldModel = CN()
    conf.Models.WorldModel.InChannels = 0
    conf.Models.WorldModel.TransformerMaxLength = 0
    conf.Models.WorldModel.TransformerHiddenDim = 0
    conf.Models.WorldModel.TransformerNumLayers = 0
    conf.Models.WorldModel.TransformerNumHeads = 0

    conf.Models.Agent = CN()
    conf.Models.Agent.NumLayers = 0
    conf.Models.Agent.HiddenDim = 256
    conf.Models.Agent.Gamma = 1.0
    conf.Models.Agent.Lambda = 0.0
    conf.Models.Agent.EntropyCoef = 0.0

    conf.JointTrainAgent = CN()
    conf.JointTrainAgent.SampleMaxSteps = 0
    conf.JointTrainAgent.BufferMaxLength = 0
    conf.JointTrainAgent.BufferWarmUp = 0
    conf.JointTrainAgent.NumEnvs = 0
    conf.JointTrainAgent.BatchSize = 0
    conf.JointTrainAgent.DemonstrationBatchSize = 0
    conf.JointTrainAgent.BatchLength = 0
    conf.JointTrainAgent.ImagineBatchSize = 0
    conf.JointTrainAgent.ImagineDemonstrationBatchSize = 0
    conf.JointTrainAgent.ImagineContextLength = 0
    conf.JointTrainAgent.ImagineBatchLength = 0
    conf.JointTrainAgent.TrainDynamicsEverySteps = 0
    conf.JointTrainAgent.TrainAgentEverySteps = 0
    conf.JointTrainAgent.SaveEverySteps = 0
    conf.JointTrainAgent.UseDemonstration = False
    
    conf.JointTrainAgent.Retrieval = CN()
    conf.JointTrainAgent.Retrieval.enable = False
    conf.JointTrainAgent.Retrieval.trigger_mode = "absolute" # "absolute" or "z_score"
    conf.JointTrainAgent.Retrieval.anchor_offset = -2
    conf.JointTrainAgent.Retrieval.hash_bits = 12
    conf.JointTrainAgent.Retrieval.threshold = 1.0
    conf.JointTrainAgent.Retrieval.z_score_threshold = 2.0
    conf.JointTrainAgent.Retrieval.ema_alpha = 0.01
    conf.JointTrainAgent.Retrieval.max_bucket_size = 100000
    conf.JointTrainAgent.Retrieval.max_anchors = 10
    conf.JointTrainAgent.Retrieval.multiplier = 5
    conf.JointTrainAgent.Retrieval.target = 5
    conf.JointTrainAgent.Retrieval.max_contexts = 256
    conf.JointTrainAgent.Retrieval.global_rebuild_enable = True
    conf.JointTrainAgent.Retrieval.global_rebuild_threshold = 0.2
    conf.JointTrainAgent.Retrieval.global_rebuild_cooldown = 2000
    conf.JointTrainAgent.Retrieval.anchor_queue_capacity = 8192

    conf.defrost()
    conf.merge_from_file(config_path)
    conf.freeze()

    return conf
