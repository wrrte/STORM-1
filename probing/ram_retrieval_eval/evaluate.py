"""Frozen-policy rollout + paired retrieval/random RAM similarity evaluation.

Copy this entire folder into another STORM-1 checkout's probing/ directory.
python probing/ram_retrieval_eval/evaluate.py --game all --device cuda:0
The RAM reader diagnostic is deliberately at the top of this file.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from copy import copy
from collections import deque, OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# RAM BINDING DIAGNOSTIC -- runs before model loading or similarity evaluation.
# Old ALE + NumPy 2 can silently return 128 copies of the first byte from
# getRAM(). The output-buffer overload can still return the correct 128 bytes.
# ---------------------------------------------------------------------------
def valid_ram(ram):
    array = np.asarray(ram)
    return (array.shape == (128,) and np.issubdtype(array.dtype, np.integer)
            and bool(np.all((array >= 0) & (array <= 255)))
            and bool(np.any(array != array[0])))


def diagnose_ram(env, samples=8):
    """Compare both APIs at identical emulator states and select a valid reader.

    Different pre-filled buffers additionally detect incomplete buffer writes.
    This advances NOOPs; the caller resets with the rollout seed afterwards.
    """
    ale, checks = env.unwrapped.ale, []
    for _ in range(samples):
        check = {}
        try:
            normal = np.asarray(ale.getRAM()).copy()
            check.update(normal_valid=valid_ram(normal), normal_unique=int(np.unique(normal).size),
                         normal_example=normal.tolist())
        except Exception as error:
            normal = None
            check.update(normal_valid=False, normal_error=f"{type(error).__name__}: {error}")
        try:
            first, second = np.full(128, 0x55, np.uint8), np.full(128, 0xAA, np.uint8)
            ale.getRAM(first)
            ale.getRAM(second)
            check.update(buffer_supported=True,
                         buffer_valid=valid_ram(first) and np.array_equal(first, second),
                         buffer_unique=int(np.unique(first).size), buffer_example=first.tolist())
        except TypeError:
            first = None
            check.update(buffer_supported=False, buffer_valid=False)
        except Exception as error:
            first = None
            check.update(buffer_supported=True, buffer_valid=False,
                         buffer_error=f"{type(error).__name__}: {error}")
        if check["normal_valid"] and check["buffer_valid"] and not np.array_equal(normal, first):
            raise RuntimeError("Both RAM APIs look nonconstant but disagree at the same emulator state")
        checks.append(check)
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            env.reset()
    buffer_ok = all(c["buffer_valid"] for c in checks)
    normal_ok = all(c["normal_valid"] for c in checks)
    method = "buffer" if buffer_ok else "normal" if normal_ok else None
    report = dict(selected_reader=method, standard_api_problem=not normal_ok,
                  samples=checks, numpy_version=np.__version__)
    if method is None:
        raise RuntimeError("Neither getRAM() nor getRAM(buffer) produced valid RAM. "
                           "Stop evaluation; use a compatible ALE/NumPy environment. " + json.dumps(report))
    return report


def read_checked_ram(ale, method):
    if method == "buffer":
        ram = np.empty(128, np.uint8)
        ale.getRAM(ram)
    elif method == "normal":
        ram = np.asarray(ale.getRAM()).copy()
    else:
        raise ValueError(f"Unknown RAM reader {method}")
    if not valid_ram(ram):
        raise RuntimeError("Corrupt RAM: expected 128 nonconstant bytes in [0,255]. Evaluation stopped.")
    return ram.astype(np.uint8, copy=False)


BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]
GAMES = ("seaquest", "frostbite", "pong")
ALE_NAMES = dict(zip(GAMES, ("Seaquest", "Frostbite", "Pong")))


def seed_all(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IsolatedRNG:
    """Analytics have a persistent RNG stream, without consuming policy RNG."""
    def __init__(self, seed, torch, device):
        self.torch, self.device = torch, device
        self.state = None
        outer = self.capture()
        seed_all(seed, torch)
        self.state = self.capture()
        self.restore(outer)

    def capture(self):
        cuda = self.torch.cuda.get_rng_state(self.device).clone() if self.device.type == "cuda" else None
        return random.getstate(), np.random.get_state(), self.torch.get_rng_state().clone(), cuda

    def restore(self, state):
        random.setstate(state[0])
        np.random.set_state(state[1])
        self.torch.set_rng_state(state[2])
        if state[3] is not None:
            self.torch.cuda.set_rng_state(state[3], self.device)

    @contextmanager
    def use(self):
        outer = self.capture()
        self.restore(self.state)
        try:
            yield
        finally:
            self.state = self.capture()
            self.restore(outer)


def import_components(root):
    # No train.py/eval.py import: no W&B init, training loop or source edits.
    sys.path.insert(0, str(root))
    import torch
    from agents import ActorCriticAgent
    from sub_models.world_models import WorldModel
    from sub_models.attention_blocks import get_subsequent_mask
    try:
        from .retrieval_snapshot import RetrievalContextManager
    except ImportError:
        from retrieval_snapshot import RetrievalContextManager
    return torch, ActorCriticAgent, WorldModel, get_subsequent_mask, RetrievalContextManager


def fingerprint(models):
    import torch
    digest = hashlib.sha256()
    for model in models:
        for name, value in model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_models(entry, checkpoint, step, device, precision, components):
    torch, Actor, World, _, _ = components
    suffix = "final" if str(step) == "final" else str(int(step))
    paths = [checkpoint / f"{kind}_{suffix}.pth" for kind in ("world_model", "agent")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{path}. Pass the correct --checkpoint/--step; no checkpoint fallback is used.")
    def load(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # Older PyTorch on the other server
            return torch.load(path, map_location="cpu")
    wm_state, agent_state = map(load, paths)
    config = entry["config"]
    w, a = config["Models"]["WorldModel"], config["Models"]["Agent"]
    action_dim = wm_state["storm_transformer.stem.0.weight"].shape[1] - 1024
    wm = World(in_channels=w["InChannels"], action_dim=action_dim,
               transformer_max_length=w["TransformerMaxLength"],
               transformer_hidden_dim=w["TransformerHiddenDim"],
               transformer_num_layers=w["TransformerNumLayers"], transformer_num_heads=w["TransformerNumHeads"]).to(device)
    agent = Actor(feat_dim=1024 + w["TransformerHiddenDim"], num_layers=a["NumLayers"],
                  hidden_dim=a["HiddenDim"], action_dim=action_dim, gamma=a["Gamma"],
                  lambd=a["Lambda"], entropy_coef=a["EntropyCoef"]).to(device)
    wm.load_state_dict(wm_state, strict=True)
    agent.load_state_dict(agent_state, strict=True)
    for model in (wm, agent):
        model.eval().requires_grad_(False)
        model.use_amp = device.type == "cuda" and precision != "fp32"
        if precision in ("bf16", "fp16"):
            model.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        model.tensor_dtype = model.amp_dtype if model.use_amp else torch.float32
    return wm, agent, action_dim, [str(path.resolve()) for path in paths]


def make_env(game, image_size=64, full_action_space=False, storm_root=ROOT):
    import gymnasium as gym
    import ale_py
    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)
    sys.path.insert(0, str(storm_root))
    from env_wrapper import MaxLast2FrameSkipWrapper
    env = gym.make(f"ALE/{ALE_NAMES[game]}-v5", frameskip=1,
                   full_action_space=full_action_space, render_mode="rgb_array")
    env = MaxLast2FrameSkipWrapper(env, skip=4)
    return gym.wrappers.ResizeObservation(env, shape=image_size)


def score_batch(replay, wm, agent, manager, rng, batch_size, score_length, components, calibrate=False):
    """Frozen equivalent of value-feature calculation + ORIGINAL trigger function."""
    torch, _, _, get_mask, _ = components
    endpoints = replay.valid_endpoints(score_length)
    if not len(endpoints):
        return 0
    endpoints = rng.choice(endpoints, size=min(batch_size, len(endpoints)), replace=False)
    indexes = (endpoints[:, None] - np.arange(score_length - 1, -1, -1)[None]) % replay.capacity
    obs = torch.from_numpy(np.asarray(replay.obs_buffer[indexes, 0])).to(manager.device).permute(0, 1, 4, 2, 3).float() / 255
    actions = torch.from_numpy(np.asarray(replay.action_buffer[indexes, 0])).to(manager.device)
    latent = wm.encode_obs(obs)
    with torch.autocast(device_type=torch.device(manager.device).type, dtype=wm.amp_dtype, enabled=wm.use_amp):
        features = wm.storm_transformer(latent, actions, get_mask(latent))
    values = agent.value(torch.cat([latent.float(), features.float()], dim=-1)).squeeze(-1)
    rewards = torch.from_numpy(np.asarray(replay.reward_buffer[indexes, 0])).to(manager.device)
    terminals = torch.from_numpy(np.asarray(replay.termination_buffer[indexes, 0])).to(manager.device)
    return manager.add_batch_transitions(values, rewards, terminals, agent.gamma,
                                         indexes[:, 0].astype(np.int64), np.zeros(len(indexes), np.int64),
                                         replay.capacity, skip_len=manager.context_length, is_warmup=calibrate)


class PairWriter:
    def __init__(self, output, game, random_repeats, random_seed, examples):
        try:
            from .metrics import GROUPS
        except ImportError:
            from metrics import GROUPS
        self.game, self.output = game, output
        self.evaluation_seed = 0
        self.state_cache = OrderedDict()
        self.random_repeats, self.rng = random_repeats, np.random.default_rng(random_seed)
        self.examples, self.queries, self.seen = examples, [], set()
        self.files, self.writers = [], {}
        fields = ["query_id", "evaluation_step", "anchor_step", "anchor_episode", "neighbor_step",
                  "neighbor_episode", "same_episode", "context_overlap", "neighbor_pointer", "repeat", "rank",
                  "distance", "similarity", *[f"distance_{g}" for g in GROUPS[game]], "valid", "error"]
        for source in ("retrieval", "random"):
            file = (output / f"{source}_neighbors.csv").open("w", newline="")
            self.files.append(file)
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            self.writers[source] = writer
        self.details = (output / "pairs.jsonl").open("w")
        self.files.append(self.details)
        self.query_stream = (output / "queries.jsonl").open("w")
        self.files.append(self.query_stream)
        query_file = (output / "queries.csv").open("w", newline="")
        self.files.append(query_file)
        self.query_writer = csv.DictWriter(query_file, fieldnames=["query_id", "evaluation_seed", "evaluation_step",
            "anchor_step", "anchor_episode", "anchor_pointer", "neighbor_count", "valid_retrieval_pairs",
            "valid_random_pairs", "retrieval_distance", "random_distance", "lazy_hash_hit_rate"])
        self.query_writer.writeheader()

    def evaluate(self, replay, manager, wm, anchor, key, evaluation_step):
        try:
            from .metrics import semantic_distance
            from .ram_states import extract_state
        except ImportError:
            from metrics import semantic_distance
            from ram_states import extract_state
        def state_at(pointer):
            step = int(replay.global_step_buffer[pointer])
            if step not in self.state_cache:
                self.state_cache[step] = extract_state(self.game, replay.ram_buffer[pointer])
                if len(self.state_cache) > 8192:
                    self.state_cache.popitem(last=False)
            self.state_cache.move_to_end(step)
            return self.state_cache[step]
        anchor_step = int(replay.global_step_buffer[anchor])
        if anchor_step in self.seen:
            return False
        self.seen.add(anchor_step)
        query_id = len(self.queries)
        query = dict(query_id=query_id, evaluation_seed=self.evaluation_seed, evaluation_step=evaluation_step, anchor_step=anchor_step,
                     anchor_episode=int(replay.episode_buffer[anchor]), anchor_pointer=int(anchor),
                     neighbor_count=0, valid_retrieval_pairs=0, valid_random_pairs=0,
                     retrieval_distance=None, random_distance=None)
        manager.active_anchors.clear()
        manager.active_anchors.append(((int(anchor), 0), int(key)))
        result = manager.retrieve_contexts(replay, wm, max_anchors=1,
                    multiplier=int(manager.config.get("multiplier", 16)), target=int(manager.config.get("target", 16)),
                    max_contexts=int(manager.config.get("max_contexts", 256)), return_indices=True)
        *_, indices = result
        neighbors = [int(p) for p, env in indices if (p, env) != (int(anchor), 0)]
        query["neighbor_count"] = len(neighbors)
        query["lazy_hash_hit_rate"] = float(result[3])
        distances = {"retrieval": [], "random": []}
        anchor_state = state_at(anchor)
        random_sets = [replay.sample_random(len(neighbors), anchor, manager.context_length, self.rng)
                       for _ in range(self.random_repeats)] if neighbors else []
        for source, repetitions in (("retrieval", [neighbors]), ("random", random_sets)):
            for repeat, selected in enumerate(repetitions):
                for rank, pointer in enumerate(selected):
                    neighbor_step = int(replay.global_step_buffer[pointer])
                    neighbor_episode = int(replay.episode_buffer[pointer])
                    row = dict(query_id=query_id, evaluation_step=evaluation_step, anchor_step=anchor_step,
                               anchor_episode=query["anchor_episode"], neighbor_step=neighbor_step,
                               neighbor_episode=neighbor_episode, same_episode=neighbor_episode == query["anchor_episode"],
                               context_overlap=bool(abs(neighbor_step - anchor_step) < manager.context_length),
                               neighbor_pointer=pointer, repeat=repeat, rank=rank, valid=True, error="")
                    try:
                        row.update(semantic_distance(self.game, anchor_state, state_at(pointer)))
                        distances[source].append(row["distance"])
                    except ValueError as error:
                        row.update(valid=False, error=str(error))
                    self.writers[source].writerow(row)
                    payload = dict(source=source, **row, anchor_ram=replay.ram_buffer[anchor].tolist(),
                                   neighbor_ram=replay.ram_buffer[pointer].tolist())
                    self.details.write(json.dumps(payload) + "\n")
        for source in distances:
            query[f"valid_{source}_pairs"] = len(distances[source])
            if distances[source]:
                query[f"{source}_distance"] = float(np.mean(distances[source]))
        # Compare complete matched groups only; do not silently drop mapping failures.
        if len(distances["retrieval"]) != len(neighbors) or len(distances["random"]) != len(neighbors) * self.random_repeats:
            query["retrieval_distance"] = query["random_distance"] = None
        self.queries.append(query)
        self.query_writer.writerow(query)
        self.query_stream.write(json.dumps(query) + "\n")
        if query_id < self.examples and neighbors:
            self.save_example(replay, query_id, anchor, neighbors, random_sets[0])
        return True

    def save_example(self, replay, query_id, anchor, retrieved, random):
        from PIL import Image, ImageDraw
        folder = self.output / "examples"
        folder.mkdir(exist_ok=True)
        n = max(len(retrieved), len(random)) + 1
        image = Image.new("RGB", (2 * 256, n * 286), "#151922")
        draw = ImageDraw.Draw(image)
        for col, (name, pointers) in enumerate((("retrieval", [anchor, *retrieved]), ("random", [anchor, *random]))):
            for row, pointer in enumerate(pointers):
                frame = Image.fromarray(np.asarray(replay.obs_buffer[pointer, 0])).resize((256, 256), Image.Resampling.NEAREST)
                image.paste(frame, (col * 256, row * 286 + 25))
                step = int(replay.global_step_buffer[pointer])
                draw.text((col * 256 + 5, row * 286 + 5), f"{name} {'ANCHOR' if row == 0 else row}: step {step}", fill="white")
        image.save(folder / f"query_{query_id:05d}.png")

    def close(self):
        for file in self.files:
            file.close()

    def flush(self):
        for file in self.files:
            file.flush()


def run_evaluation(args, entry, output):
    game, config = entry["game"], entry["config"]
    output.mkdir(parents=True, exist_ok=False)
    try:
        from .replay import EvaluationReplay
        from .metrics import summarize_queries
    except ImportError:
        from replay import EvaluationReplay
        from metrics import summarize_queries
    image_size = int(config["BasicSettings"]["ImageSize"])
    env = make_env(game, image_size, storm_root=args.storm_root)
    writer = replay = None
    metadata = dict(game=game, run_id=entry.get("run_id"), wandb_url=entry.get("wandb_url"),
                    requested_steps=args.steps, evaluation_seed=args.seed, config=config,
                    protocol="online causal replay; frozen learned policy; original retrieval class snapshot; random is measurement-only",
                    scores="current visible semantic state; excludes temperature, velocity, direction and cumulative score",
                    optimizer_updates=0, distinct_anchor_steps=True, status="running",
                    metric_sha256=hashlib.sha256((BUNDLE / "metrics.py").read_bytes()).hexdigest(),
                    retrieval_sha256=hashlib.sha256((BUNDLE / "retrieval_snapshot.py").read_bytes()).hexdigest())
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    try:
        env.reset(seed=args.seed)
        try:
            ram_check = diagnose_ram(env)
        except RuntimeError as error:
            (output / "ram_check.json").write_text(json.dumps(dict(status="failed", error=str(error)), indent=2))
            raise
        import ale_py
        import gymnasium
        ram_check.update(ale_version=ale_py.__version__, gymnasium_version=gymnasium.__version__)
        (output / "ram_check.json").write_text(json.dumps(ram_check, indent=2))
        print(f"[{game}] RAM check: standard_api_problem={ram_check['standard_api_problem']}, "
              f"selected={ram_check['selected_reader']}", flush=True)
        if args.ram_check_only:
            metadata["status"] = "ram_check_only"
            return metadata
        components = import_components(args.storm_root)
        torch, _, _, _, Manager = components
        torch.set_num_threads(args.cpu_threads)
        device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        seed_all(args.seed, torch)
        checkpoint = args.checkpoint or args.checkpoint_root / entry["checkpoint"]
        step = args.step or entry.get("step", "final")
        wm, agent, action_dim, paths = load_models(entry, checkpoint, step, device, args.precision, components)
        if env.action_space.n != action_dim:
            env.close()
            env = make_env(game, image_size, full_action_space=True, storm_root=args.storm_root)
            if env.action_space.n != action_dim:
                raise ValueError(f"Environment action count {env.action_space.n} != model {action_dim}")
        before_hash = fingerprint((wm, agent))
        metadata.update(checkpoint_paths=paths, checkpoint_step=str(step), model_hash_before=before_hash,
                        device=str(device), precision=args.precision, torch_version=torch.__version__,
                        effective_model_dtype=str(wm.amp_dtype) if wm.use_amp else "torch.float32")
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        analytic_rng = IsolatedRNG(args.seed + 10001, torch, device)
        sampler = np.random.default_rng(args.seed + 10002)
        retrieval_config = dict(config["JointTrainAgent"]["Retrieval"])
        if retrieval_config.get("neighbor_source", "retrieval") != "retrieval":
            raise ValueError("This evaluator expects a retrieval-trained run, not a random-neighbor preset")
        retrieval_config.update(enable=True, save_warmup=False,
                                context_length=int(config["JointTrainAgent"]["ImagineContextLength"]))
        with analytic_rng.use():
            manager = Manager(1, retrieval_config, 1024, device=str(device))
        capacity = min(args.steps, args.buffer_size or int(config["JointTrainAgent"]["BufferMaxLength"]))
        score_length = int(config["JointTrainAgent"]["BatchLength"])
        if not manager.context_length + 2 <= score_length <= config["Models"]["WorldModel"]["TransformerMaxLength"]:
            raise ValueError("Invalid saved context lengths")
        if args.index_warmup < max(score_length, manager.hash_bits + 1) or args.index_warmup >= capacity:
            raise ValueError("index-warmup must cover BatchLength/PCA rank and be smaller than steps/buffer size")
        replay = EvaluationReplay(capacity, image_size, output / "replay")
        metadata["replay_capacity"] = capacity
        writer = PairWriter(output, game, args.random_repeats, args.seed + 20000, args.examples)
        writer.evaluation_seed = args.seed
        obs, info = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        lives = info.get("lives", int(env.unwrapped.ale.lives()))
        contexts, actions = deque(maxlen=16), deque(maxlen=16)
        episode = episode_step = life_id = 0
        episode_return, completed_returns, counters = 0., [], dict(triggered_anchors=0, duplicate_anchors=0, query_rounds=0)
        indexed = False
        started = time.monotonic()
        with torch.inference_mode():
            for step_index in range(args.steps):
                current = torch.from_numpy(np.asarray(obs).copy()).to(device).permute(2, 0, 1)[None, None].float() / 255
                if not actions:
                    action = int(env.action_space.sample())
                else:
                    latent = wm.encode_obs(torch.cat(list(contexts), dim=1))
                    action_context = torch.as_tensor(np.asarray(actions)[None], device=device)
                    prior, features = wm.calc_last_dist_feat(latent, action_context)
                    action = int(np.asarray(agent.sample_as_env_action(torch.cat([prior, features], dim=-1), greedy=False)).item())
                ram = read_checked_ram(env.unwrapped.ale, ram_check["selected_reader"])
                contexts.append(current)
                actions.append(action)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                new_lives = next_info.get("lives", int(env.unwrapped.ale.lives()))
                life_lost = new_lives < lives
                pointer = replay.append(obs, ram, action, reward, bool(terminated or truncated or life_lost),
                                        step_index, episode, episode_step, life_id)
                episode_return += float(reward)
                episode_step += 1
                if indexed:
                    with analytic_rng.use():
                        latent_for_hash = wm.encode_obs(current, sample_mode=manager.hash_sample_mode).squeeze(1)
                        manager.add_transition(pointer, -1, latent_for_hash)
                if step_index + 1 == args.index_warmup:
                    with analytic_rng.use():
                        manager.rebuild_all_hash_buckets(replay, wm, chunk_size=args.encode_chunk)
                        if manager.use_pca and manager.hash_mean is None:
                            raise RuntimeError("PCA did not produce a valid projection")
                        for _ in range(args.calibration_batches):
                            score_batch(replay, wm, agent, manager, sampler, args.score_batch_size,
                                        score_length, components, calibrate=True)
                    metadata["ema_initialization"] = "original is_warmup=True EMA calibration on fresh rollout; not historical saved EMA"
                    metadata["ema_after_calibration"] = {name: [x.tolist() for x in manager.get_ema_stats(name)]
                                                         for name in manager.statistics_signals}
                    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
                    indexed = True
                    print(f"[{game}] indexed {replay.length} policy frames; calibrated EMA", flush=True)
                elif indexed and args.rebuild_every and (step_index + 1 - args.index_warmup) % args.rebuild_every == 0:
                    with analytic_rng.use():
                        manager.rebuild_all_hash_buckets(replay, wm, chunk_size=args.encode_chunk)
                if indexed and (step_index + 1 - args.index_warmup) % args.query_every == 0:
                    with analytic_rng.use():
                        counters["query_rounds"] += 1
                        counters["triggered_anchors"] += score_batch(replay, wm, agent, manager, sampler,
                                                    args.score_batch_size, score_length, components)
                        anchors = list(manager.active_anchors)[:int(manager.config.get("max_anchors", 16))]
                        manager.active_anchors.clear()
                        for (anchor, env_id), key in anchors:
                            if not manager._valid_context(replay, anchor, env_id):
                                continue
                            if not writer.evaluate(replay, manager, wm, anchor, key, step_index):
                                counters["duplicate_anchors"] += 1
                obs, lives = next_obs, new_lives
                if life_lost:
                    life_id += 1
                if terminated or truncated:
                    completed_returns.append(episode_return)
                    episode, episode_step, episode_return = episode + 1, 0, 0.
                    life_id += 1
                    contexts.clear()
                    actions.clear()
                    obs, info = env.reset()
                    lives = info.get("lives", int(env.unwrapped.ale.lives()))
                if (step_index + 1) % args.log_every == 0:
                    writer.flush()
                    progress = summarize_queries(writer.queries, seed=args.seed, bootstrap_samples=min(1000, args.bootstrap_samples))
                    progress.update(status="running", steps_so_far=step_index + 1)
                    (output / "progress_summary.json").write_text(json.dumps(progress, indent=2))
                    print(f"[{game}] steps={step_index + 1}/{args.steps}, episodes={episode}, "
                          f"unique_queries={len(writer.queries)}, elapsed={time.monotonic() - started:.1f}s", flush=True)
        after_hash = fingerprint((wm, agent))
        metadata.update(model_hash_after=after_hash, parameters_unchanged=before_hash == after_hash,
                        completed_episode_returns=completed_returns, partial_episode_return=episode_return,
                        counters=counters, replay_length=replay.length, last_pointer=replay.last_pointer,
                        actual_steps=args.steps, retrieval_config=retrieval_config,
                        analysis_settings={k: getattr(args, k) for k in ("index_warmup", "query_every", "score_batch_size",
                            "calibration_batches", "rebuild_every", "random_repeats", "buffer_size", "seed")},
                        elapsed_seconds=time.monotonic() - started)
        if before_hash != after_hash:
            raise RuntimeError("Model weights/buffers changed during frozen evaluation")
        summary = summarize_queries(writer.queries, seed=args.seed, bootstrap_samples=args.bootstrap_samples)
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        (output / "queries.json").write_text(json.dumps(writer.queries, indent=2))
        metadata["status"] = "complete" if summary["status"] == "ok" else "complete_no_valid_pairs"
        print(f"[{game}] {json.dumps(summary)}", flush=True)
        return metadata
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if writer:
            writer.close()
        if replay:
            replay.flush()
        env.close()
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=("all", *GAMES), default="all", type=str.lower)
    parser.add_argument("--manifest", type=Path, default=BUNDLE / "runs_9999.json")
    parser.add_argument("--storm-root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Override checkpoint directory for one game")
    parser.add_argument("--config", type=Path, help="Override config YAML for one game")
    parser.add_argument("--step", help="final (default) or an explicit saved checkpoint step")
    parser.add_argument("--output", type=Path, default=ROOT / "results/ram_verification/retrieval_9999")
    parser.add_argument("--ram-check-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--seeds", type=int, nargs="+", help="Independent rollout seeds; aggregate per game after all runs")
    parser.add_argument("--steps", type=int, default=20000, help="Policy decisions per game, 4 ALE frames per decision")
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--index-warmup", type=int, default=2048)
    parser.add_argument("--query-every", type=int, default=32)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--calibration-batches", type=int, default=64)
    parser.add_argument("--rebuild-every", type=int, default=2048, help="0 keeps the initial evaluation PCA fixed")
    parser.add_argument("--encode-chunk", type=int, default=128)
    parser.add_argument("--random-repeats", type=int, default=1)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args(argv)
    for name in ("steps", "query_every", "score_batch_size", "index_warmup", "encode_chunk", "random_repeats", "log_every", "cpu_threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if min(args.calibration_batches, args.examples, args.rebuild_every, args.bootstrap_samples) < 0:
        parser.error("calibration-batches/examples/rebuild-every must be nonnegative")
    if args.buffer_size is not None and args.buffer_size < 1:
        parser.error("buffer-size must be positive")
    if not 0 <= args.seed < 2**32 - 20001:
        parser.error("seed is outside the supported range")
    if args.seeds and (len(args.seeds) != len(set(args.seeds)) or any(not 0 <= s < 2**32 - 20001 for s in args.seeds)):
        parser.error("--seeds must contain distinct valid rollout seeds")
    if args.game == "all" and (args.checkpoint or args.config):
        parser.error("--checkpoint/--config overrides require a single --game")
    args.checkpoint_root = args.checkpoint_root or args.storm_root / "ckpt"
    return args


def main(argv=None):
    args = parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    entries = [entry for entry in manifest["runs"] if args.game in ("all", entry["game"])]
    if not entries:
        raise ValueError("No matching game in manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        entry = dict(entry)
        if args.config:
            import yaml
            entry["config"] = yaml.safe_load(args.config.read_text())
        if args.checkpoint:
            entry["preset_run_id"], entry["run_id"], entry["wandb_url"] = entry.get("run_id"), None, None
        destinations = []
        for seed in args.seeds or [args.seed]:
            local_args = copy(args)
            local_args.seed = seed
            destination = args.output / entry["game"]
            if args.seeds:
                destination /= f"seed_{seed}"
            if destination.exists():
                raise FileExistsError(f"{destination} exists; choose a new --output to preserve previous measurements")
            run_evaluation(local_args, entry, destination)
            if args.seeds and not args.ram_check_only:
                destinations.append(destination)
        if args.seeds and not args.ram_check_only:
            try:
                from .aggregate import aggregate
            except ImportError:
                from aggregate import aggregate
            summary = aggregate(destinations, seed=args.seed, bootstrap_samples=args.bootstrap_samples)
            (args.output / entry["game"] / "summary_all_seeds.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
