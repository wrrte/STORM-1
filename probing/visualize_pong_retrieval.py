"""Offline Pong retrieval analysis. No environment rollout or parameter updates.

Run from STORM-1 (see PONG_RETRIEVAL_VISUALIZATION.md)::

    python probing/visualize_pong_retrieval.py \
        --checkpoint ckpt/Pong-9999_Shared --output results/pong_retrieval

The scores and retrieval are newly computed from a frozen checkpoint. They are
not a reconstruction of historical training queries. Images come directly from
the saved replay observations, not the world model's image decoder.
"""

import argparse
import csv
import json
import random
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml

STORM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STORM_ROOT.parent))
sys.path.insert(0, str(STORM_ROOT))

from agents import ActorCriticAgent
from retrieval import RetrievalContextManager
from sub_models.attention_blocks import get_subsequent_mask
from sub_models.world_models import WorldModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-context-length", type=int, default=None,
                        help="Defaults to the saved BatchLength (64 for this checkpoint)")
    parser.add_argument("--post-score-steps", type=int, default=3,
                        help="Consider the positive-reward transition and this many subsequent transitions")
    parser.add_argument("--latent-mode", choices=("random_sample", "probs"), default="random_sample")
    parser.add_argument("--anchor-offset", type=int, default=None,
                        help="Defaults to saved offset; anchor = surprise transition + 1 + offset")
    parser.add_argument("--target", type=int, default=None,
                        help="Retrieved contexts INCLUDING anchor; defaults to checkpoint config")
    parser.add_argument("--multiplier", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None,
                        help="Override the saved surprise trigger threshold for selection")
    parser.add_argument("--reward-pointer", type=int, default=None,
                        help="Restrict selection to this positive-reward replay pointer")
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument("--action-labels", nargs="+", default=None,
                        help="Optional labels in action-ID order; otherwise show recorded numeric IDs")
    args = parser.parse_args()
    if args.batch_size < 1 or args.post_score_steps < 0:
        parser.error("batch-size must be positive and post-score-steps nonnegative")
    if args.target is not None and args.target < 2:
        parser.error("target must be at least 2 to display a neighbor")
    if args.multiplier is not None and args.multiplier < 1:
        parser.error("multiplier must be positive")
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint(path):
    path = path.resolve()
    if (path / "warmup.json").is_file():
        locator = json.loads((path / "warmup.json").read_text())
        path = path / locator["checkpoint"]
    for name in ("config.yaml", "training_state.pt", "replay_buffer.npz"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    return path


class SavedReplay:
    """CPU replay view without allocating the unused portion of the buffer."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            for name in ("obs", "action", "reward", "termination"):
                setattr(self, name + "_buffer", data[name])
            for name in ("length", "last_pointer", "num_envs", "max_length"):
                setattr(self, name, int(data[name]))
        self.store_on_gpu = False
        self.capacity = self.max_length // self.num_envs
        self.length = min(self.length, self.capacity)
        self.oldest = (self.last_pointer + 1) % self.capacity if self.length == self.capacity else 0

    def pointer(self, position):
        return int((self.oldest + position) % self.capacity)

    def position(self, pointer):
        return int((pointer - self.oldest) % self.capacity)

    def window(self, first, last):
        return (np.arange(first, last + 1, dtype=np.int64) + self.oldest) % self.capacity

    def next_observation(self, pointer, env):
        # obs[p+1] after a terminal transition can be an autoreset observation.
        if self.termination_buffer[pointer, env] > .5:
            return None, "terminal transition: final observation was not stored separately"
        if self.position(pointer) + 1 >= self.length:
            return None, "latest replay transition: next observation was not stored"
        return self.obs_buffer[(pointer + 1) % self.capacity, env], None


def load_analysis(args):
    checkpoint = resolve_checkpoint(args.checkpoint)
    config = yaml.safe_load((checkpoint / "config.yaml").read_text())
    metadata_path = checkpoint / "warmup_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if metadata.get("env_name") and "Pong" not in metadata["env_name"]:
        raise ValueError("This analysis selects positive-reward events in Pong")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    args.seed = metadata.get("seed", 0) if args.seed is None else args.seed
    seed_all(args.seed)
    # The local checkpoint contains FastHashBucket objects as well as tensors.
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    replay = SavedReplay(checkpoint / "replay_buffer.npz")
    if not 0 <= args.env_index < replay.num_envs:
        raise ValueError("env-index is outside the saved replay's environment range")
    wm_config, agent_config = config["Models"]["WorldModel"], config["Models"]["Agent"]
    latent_dim = 32 * 32
    action_dim = state["world_model_state_dict"]["storm_transformer.stem.0.weight"].shape[1] - latent_dim
    world_model = WorldModel(
        in_channels=wm_config["InChannels"], action_dim=action_dim,
        transformer_max_length=wm_config["TransformerMaxLength"],
        transformer_hidden_dim=wm_config["TransformerHiddenDim"],
        transformer_num_layers=wm_config["TransformerNumLayers"],
        transformer_num_heads=wm_config["TransformerNumHeads"],
    ).to(device)
    agent = ActorCriticAgent(
        feat_dim=latent_dim + wm_config["TransformerHiddenDim"],
        num_layers=agent_config["NumLayers"], hidden_dim=agent_config["HiddenDim"],
        action_dim=action_dim, gamma=agent_config["Gamma"], lambd=agent_config["Lambda"],
        entropy_coef=agent_config["EntropyCoef"],
    ).to(device)
    world_model.load_state_dict(state["world_model_state_dict"])
    agent.load_state_dict(state["agent_state_dict"])
    for model in (world_model, agent):
        model.eval().requires_grad_(False)
        model.use_amp = device.type == "cuda"
    retrieval_config = dict(config["JointTrainAgent"]["Retrieval"])
    retrieval_config.update(enable=True, save_warmup=False,
                            context_length=config["JointTrainAgent"]["ImagineContextLength"])
    for key in ("anchor_offset", "target", "multiplier"):
        value = getattr(args, key)
        if value is not None:
            retrieval_config[key] = value
    manager = RetrievalContextManager(replay.num_envs, retrieval_config, latent_dim, device=str(device))
    if state.get("retrieval_state") is None:
        raise ValueError("Checkpoint has no retrieval state; saved hash buckets and EMA statistics are required")
    manager.load_state_dict(state["retrieval_state"])
    # Only the newly selected analysis query is submitted to retrieve_contexts.
    manager.active_anchors.clear()
    score_length = args.score_context_length or config["JointTrainAgent"]["BatchLength"]
    if not manager.context_length + 2 <= score_length <= wm_config["TransformerMaxLength"]:
        raise ValueError("score-context-length must be between ImagineContextLength + 2 and TransformerMaxLength")
    total_steps = int(state["total_steps"])
    del state  # Optimizer states are not loaded or used by this analysis.
    return SimpleNamespace(checkpoint=checkpoint, config=config, metadata=metadata, device=device,
                           replay=replay, world_model=world_model, agent=agent, manager=manager,
                           score_length=score_length, total_steps=total_steps)


def collect_candidates(ctx, args):
    replay, manager, env = ctx.replay, ctx.manager, args.env_index
    order = replay.window(0, replay.length - 1)
    rewards = replay.reward_buffer[order, env]
    candidates = []
    for reward_position in np.flatnonzero(rewards > 0):
        reward_pointer = replay.pointer(reward_position)
        if args.reward_pointer is not None and reward_pointer != args.reward_pointer:
            continue
        for delay in range(args.post_score_steps + 1):
            position = int(reward_position + delay)
            first = position - ctx.score_length + 2
            if first < 0 or position + 1 >= replay.length:
                continue
            sequence = replay.window(first, position + 1)
            # Never use a reset observation as the bootstrap state or history.
            if np.any(replay.termination_buffer[sequence[:-1], env] > .5):
                continue
            anchor_position = position + 1 + manager.anchor_offset
            if not 0 <= anchor_position < replay.length:
                continue
            anchor_pointer = replay.pointer(anchor_position)
            if not manager._valid_context(replay, anchor_pointer, env):
                continue
            anchor_key = manager.index_to_bucket.get((anchor_pointer, env))
            if anchor_key is None:
                continue
            candidates.append(dict(
                reward_pointer=reward_pointer, reward_position=int(reward_position),
                surprise_pointer=replay.pointer(position), surprise_position=position,
                steps_after_reward=delay, anchor_pointer=anchor_pointer,
                anchor_position=anchor_position, hash_key=int(anchor_key), env=env,
                score_context_start=replay.pointer(first),
            ))
    return candidates


@torch.inference_mode()
def score_candidates(ctx, args, candidates):
    """Use the value features from WorldModel.update, without calling update.

    s is the scored transition. Input ends at s+1, so the last three values give
    V[s-1], V[s], V[s+1]. The saved EMA is held fixed across all candidates.
    """
    wm, manager, replay = ctx.world_model, ctx.manager, ctx.replay
    seed_all(args.seed)
    for start in range(0, len(candidates), args.batch_size):
        batch = candidates[start:start + args.batch_size]
        indexes = [replay.window(row["surprise_position"] - ctx.score_length + 2,
                                 row["surprise_position"] + 1) for row in batch]
        obs = np.stack([replay.obs_buffer[ix, row["env"]] for ix, row in zip(indexes, batch)])
        actions = np.stack([replay.action_buffer[ix, row["env"]] for ix, row in zip(indexes, batch)])
        obs = torch.from_numpy(obs).to(ctx.device).permute(0, 1, 4, 2, 3).float() / 255.
        actions = torch.from_numpy(actions).to(ctx.device)
        with torch.autocast(device_type=ctx.device.type, dtype=wm.amp_dtype, enabled=wm.use_amp):
            latent = wm.encode_obs(obs, sample_mode=args.latent_mode)
            dist_feat = wm.storm_transformer(latent, actions, get_subsequent_mask(latent))
        # train_world_model_step also evaluates the critic outside the WM AMP block.
        values = ctx.agent.value(torch.cat([latent.float(), dist_feat.float()], dim=-1))
        previous, current, following = values[:, -3], values[:, -2], values[:, -1]
        reward = torch.tensor([replay.reward_buffer[r["surprise_pointer"], r["env"]] for r in batch],
                              device=ctx.device)
        done = torch.tensor([replay.termination_buffer[r["surprise_pointer"], r["env"]] for r in batch],
                            device=ctx.device)
        td_error = (reward + ctx.agent.gamma * (1. - done) * following - current).abs()
        difference = current - previous
        signal = difference if manager.value_signal == "value_diff" else current
        envs = torch.tensor([r["env"] for r in batch], device=ctx.device)
        z_td = manager._z_score(td_error[:, None], "td_error", envs).squeeze(1)
        z_value = manager._z_score(signal[:, None], manager.value_signal, envs).squeeze(1)
        if manager.trigger_mode == "z_score":
            if manager.score_combination == "add":
                scores = F.relu(z_td) + F.softplus(z_value)
            else:
                scores = F.relu(z_td) * F.softplus(z_value)
        else:
            scores = td_error
        fields = dict(score=scores, abs_td_error=td_error, value_previous=previous,
                      value=current, value_next=following, value_diff=difference,
                      z_td=z_td, z_value_signal=z_value, reward_at_surprise=reward)
        for name, tensor in fields.items():
            for row, value in zip(batch, tensor.float().cpu().tolist()):
                row[name] = float(value)
        print(f"Scored {min(start + len(batch), len(candidates))}/{len(candidates)} candidates", flush=True)


def selection_threshold(manager, args):
    if args.min_score is not None:
        return args.min_score
    if manager.trigger_mode != "z_score":
        return manager.threshold
    return manager.additive_z_score_threshold if manager.score_combination == "add" else manager.z_score_threshold


@torch.inference_mode()
def retrieve_one(ctx, args, candidates, threshold):
    """Select the highest-scoring candidate that actually returns a neighbor."""
    manager = ctx.manager
    target = int(manager.config.get("target", 5))
    multiplier = int(manager.config.get("multiplier", 5))
    max_contexts = int(manager.config.get("max_contexts", 256))
    if target < 2 or multiplier < 1 or max_contexts < 2:
        raise ValueError("Retrieval needs target >= 2, multiplier >= 1 and max_contexts >= 2")
    attempts = []
    for row in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if not np.isfinite(row["score"]) or row["score"] < threshold:
            continue
        # Isolate retrieval randomness from how many score batches were computed.
        seed_all(args.seed)
        manager.active_anchors = deque([((row["anchor_pointer"], row["env"]), row["hash_key"])])
        result = manager.retrieve_contexts(
            ctx.replay, ctx.world_model, max_anchors=1, multiplier=multiplier,
            target=target, max_contexts=max_contexts, return_indices=True,
        )
        observations, actions, before_cap, hit_rate, weights, valid_anchors, indices = result
        attempts.append(dict(anchor_pointer=row["anchor_pointer"], surprise_pointer=row["surprise_pointer"],
                             score=row["score"], contexts_returned=len(indices), hit_rate=hit_rate))
        if observations is not None and len(indices) >= 2:
            return row, indices, weights, dict(candidates_before_cap=before_cap,
                lazy_hash_hit_rate=hit_rate, valid_anchors=valid_anchors, attempts=attempts)
    return None, [], [], dict(attempts=attempts)


def transition_record(ctx, pointer, env, role, weight=None):
    replay = ctx.replay
    following, unavailable = replay.next_observation(pointer, env)
    return dict(role=role, pointer=int(pointer), env=int(env),
                replay_position=replay.position(pointer),
                approximate_env_step=ctx.total_steps // replay.num_envs - replay.length + replay.position(pointer),
                action=int(replay.action_buffer[pointer, env]),
                reward=float(replay.reward_buffer[pointer, env]),
                termination=bool(replay.termination_buffer[pointer, env] > .5),
                weight=None if weight is None else float(weight),
                next_pointer=None if following is None else int((pointer + 1) % replay.capacity),
                next_observation_unavailable=unavailable)


def save_raw_transitions(ctx, indices, weights, output):
    replay, length = ctx.replay, ctx.manager.context_length
    records, obs, following, present, contexts, context_actions, context_indices = [], [], [], [], [], [], []
    for rank, ((pointer, env), weight) in enumerate(zip(indices, weights)):
        records.append(transition_record(ctx, pointer, env, "anchor" if rank == 0 else f"neighbor {rank}", weight))
        obs.append(replay.obs_buffer[pointer, env])
        next_obs, _ = replay.next_observation(pointer, env)
        present.append(next_obs is not None)
        following.append(np.zeros_like(obs[-1]) if next_obs is None else next_obs)
        ix = (pointer - np.arange(length - 1, -1, -1)) % replay.capacity
        context_indices.append(ix)
        contexts.append(replay.obs_buffer[ix, env])
        context_actions.append(replay.action_buffer[ix, env])
    np.savez_compressed(output / "transitions.npz", obs=np.stack(obs), next_obs=np.stack(following),
        next_obs_valid=np.asarray(present), action=np.asarray([r["action"] for r in records]),
        reward=np.asarray([r["reward"] for r in records]), termination=np.asarray([r["termination"] for r in records]),
        indices=np.asarray(indices), weights=np.asarray(weights), context_obs=np.stack(contexts),
        context_action=np.stack(context_actions), context_indices=np.stack(context_indices))
    return records


def action_text(action, labels):
    if labels is not None and 0 <= action < len(labels):
        return f"a = {action}\n{labels[action]}"
    return f"a = {action}"


def save_figures(ctx, args, selected, records, candidates, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    replay, output = ctx.replay, args.output
    count = len(records)
    fig, axes = plt.subplots(count, 3, figsize=(9, 1.85 * count + 1.2), squeeze=False,
                             gridspec_kw={"width_ratios": [1, .85, 1]})
    for row, record in enumerate(records):
        pointer, env = record["pointer"], record["env"]
        axes[row, 0].imshow(replay.obs_buffer[pointer, env], interpolation="nearest")
        axes[row, 0].set_title(f"{record['role']} | p={pointer}, env={env}", fontsize=10)
        axes[row, 1].text(.5, .62, action_text(record["action"], args.action_labels),
                          ha="center", va="center", fontsize=13, transform=axes[row, 1].transAxes)
        axes[row, 1].text(.5, .30, f"reward = {record['reward']:+g}\nweight = {record['weight']:.4f}\n-->",
                          ha="center", va="center", fontsize=10, transform=axes[row, 1].transAxes)
        next_obs, unavailable = replay.next_observation(pointer, env)
        if next_obs is None:
            axes[row, 2].text(.5, .5, "Next observation unavailable\n(see analysis.json)",
                              ha="center", va="center", transform=axes[row, 2].transAxes)
        else:
            axes[row, 2].imshow(next_obs, interpolation="nearest")
            axes[row, 2].set_title(f"after recorded action | p={record['next_pointer']}", fontsize=10)
        for axis in axes[row]:
            axis.axis("off")
    fig.suptitle(f"Pong: offline checkpoint retrieval | surprise={selected['score']:.3f}\n"
                 f"reward p={selected['reward_pointer']} | surprise p={selected['surprise_pointer']} | "
                 f"anchor p={selected['anchor_pointer']}\n"
                 "Saved replay observation  ->  recorded action  ->  next replay observation", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - .85 / (1.85 * count + 1.2)))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"neighbors.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Keep the scoring event distinct from the (possibly earlier) retrieval anchor.
    first = max(0, min(selected["anchor_position"], selected["reward_position"]) - 1)
    last = min(replay.length - 1, max(selected["surprise_position"] + 1,
                                    selected["reward_position"] + args.post_score_steps + 1))
    positions = list(range(first, last + 1))
    columns = min(7, len(positions))
    rows = (len(positions) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(2.3 * columns, 2.65 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for position, axis in zip(positions, axes.flat):
        pointer, env = replay.pointer(position), selected["env"]
        labels = []
        if position == selected["anchor_position"]:
            labels.append("ANCHOR")
        if position == selected["reward_position"]:
            labels.append("SCORING ACTION")
        if position == selected["reward_position"] + 1:
            labels.append("AFTER POINT")
        if position == selected["surprise_position"]:
            labels.append("SURPRISE TRANSITION")
        axis.imshow(replay.obs_buffer[pointer, env], interpolation="nearest")
        axis.set_title(f"p={pointer} | offset={position-selected['reward_position']:+d}\n"
                       f"a={int(replay.action_buffer[pointer, env])}, "
                       f"r={replay.reward_buffer[pointer, env]:+g}\n" + " / ".join(labels), fontsize=9)
    fig.suptitle("Point event and retrieval anchor (r[p] follows the action from obs[p])", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .92))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"point_event.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    related = sorted([r for r in candidates if r["reward_pointer"] == selected["reward_pointer"]
                      and r["env"] == selected["env"]], key=lambda r: r["steps_after_reward"])
    fig, axis = plt.subplots(figsize=(7, 3.6))
    axis.plot([r["steps_after_reward"] for r in related], [r["score"] for r in related], "o-")
    axis.scatter([selected["steps_after_reward"]], [selected["score"]], color="red", zorder=3, label="Selected")
    axis.axhline(threshold, linestyle="--", color="gray", label=f"Selection threshold = {threshold:g}")
    axis.set(xlabel="Transition offset from positive reward (0 = scoring action)",
             ylabel="Recomputed surprise score", title="Frozen checkpoint and saved EMA statistics")
    axis.set_xticks([r["steps_after_reward"] for r in related])
    axis.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"surprise_scores.{suffix}", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    # Fail early for missing plotting dependencies, before expensive inference.
    import matplotlib  # noqa: F401
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {args.output}")
    ctx = load_analysis(args)
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = collect_candidates(ctx, args)
    if not candidates:
        raise RuntimeError("No positive-reward candidates with intact scoring/retrieval history were found")
    score_candidates(ctx, args, candidates)
    with (args.output / "candidate_scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(sorted(candidates, key=lambda r: r["score"], reverse=True))
    threshold = selection_threshold(ctx.manager, args)
    selected, indices, weights, retrieval_info = retrieve_one(ctx, args, candidates, threshold)
    report = dict(
        provenance="offline checkpoint recomputation; not historical training retrieval",
        checkpoint=str(ctx.checkpoint), checkpoint_step=ctx.total_steps, seed=args.seed,
        device=str(ctx.device), amp=ctx.world_model.use_amp, amp_dtype=str(ctx.world_model.amp_dtype),
        scoring_model_mode="eval", latent_mode=args.latent_mode, batch_size=args.batch_size,
        score_context_length=ctx.score_length, selection_threshold=threshold,
        post_score_steps=args.post_score_steps,
        selection="highest-scoring eligible positive-reward candidate with at least one returned neighbor",
        score_definition={"td_error": "abs(r[s] + gamma * (1-done[s]) * V[s+1] - V[s])",
                          "value_diff": "V[s] - V[s-1]",
                          "z_score_multiply": "relu(z_td_error) * softplus(z_value_signal)",
                          "z_score_add": "relu(z_td_error) + softplus(z_value_signal)",
                          "absolute": "abs_td_error"},
        ema_policy="saved checkpoint statistics held fixed; never updated",
        ema={name: {"mean": mean.tolist(), "variance": var.tolist()}
             for name in ("td_error", "value_diff", "value")
             for mean, var in [ctx.manager.get_ema_stats(name)]},
        retrieval_config=dict(ctx.manager.config), saved_config=ctx.config,
        anchor_rule="anchor_pointer = surprise_pointer + 1 + anchor_offset (in replay chronology)",
        retrieval_method="original retrieve_contexts: saved hash bucket, random candidates, lazy hash recheck, final cap",
        observation_source="saved uint8 replay observations (resized, frame-skipped/max-pooled); not raw ALE resolution",
        next_observation_policy="adjacent saved observation; unavailable at terminal/newest transition",
        candidate_count=len(candidates), selected=selected, retrieval=retrieval_info,
        action_labels=args.action_labels,
    )
    if selected is None:
        (args.output / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError("No candidate above threshold returned a neighbor. See candidate_scores.csv and "
                           "analysis.json; use a new output directory when changing --min-score or --multiplier")
    report["transitions"] = save_raw_transitions(ctx, indices, weights, args.output)
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_figures(ctx, args, selected, report["transitions"], candidates, threshold)
    print(f"Saved {len(indices)-1} neighbors and anchor to {args.output}")
    print(f"Positive reward p={selected['reward_pointer']}; surprise p={selected['surprise_pointer']} "
          f"score={selected['score']:.6g}; anchor p={selected['anchor_pointer']}")


if __name__ == "__main__":
    main()
