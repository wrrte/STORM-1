"""Distances between visible semantic states; no velocity, temperature or score."""
import math

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_assignment
except ImportError:
    _scipy_assignment = None

try:
    from .ram_states import extract_state
except ImportError:
    from ram_states import extract_state


GROUPS = {
    "seaquest": ("player", "enemies", "divers", "collected_divers", "oxygen"),
    "frostbite": ("player", "enemies", "floes", "igloo_stage"),
    "pong": ("ball", "player", "enemy"),
}


def assignment_cost(costs):
    """Exact square minimum-cost assignment; NumPy fallback needs no SciPy.

    Successive augmenting paths with row/column potentials (Hungarian method).
    SciPy, when installed, provides a faster backend for long evaluation runs.
    """
    costs = np.asarray(costs, dtype=np.float64)
    n = len(costs)
    if costs.shape != (n, n) or not np.isfinite(costs).all():
        raise ValueError("Expected a finite square assignment matrix")
    if not n:
        return 0.
    if _scipy_assignment is not None:
        rows, cols = _scipy_assignment(costs)
        return float(costs[rows, cols].sum())
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    matching, previous = np.zeros(n + 1, np.int64), np.zeros(n + 1, np.int64)
    for row in range(1, n + 1):
        matching[0], col = row, 0
        minimum, used = np.full(n + 1, np.inf), np.zeros(n + 1, bool)
        while True:
            used[col] = True
            current = costs[matching[col] - 1] - u[matching[col]] - v[1:]
            improve = ~used[1:] & (current < minimum[1:])
            minimum[1:][improve] = current[improve]
            previous[1:][improve] = col
            col_next = int(np.argmin(np.where(used[1:], np.inf, minimum[1:]))) + 1
            delta = minimum[col_next]
            u[matching[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            col = col_next
            if matching[col] == 0:
                break
        while col:
            last = previous[col]
            matching[col] = matching[last]
            col = int(last)
    return float(costs[matching[1:] - 1, np.arange(n)].sum())


def object_distance(a, b, circular_x=False):
    if a["kind"] != b["kind"]:
        return 1.0
    ax, ay, aw, ah = a["xywh"]
    bx, by, bw, bh = b["xywh"]
    dx = abs((ax + aw / 2) - (bx + bw / 2))
    if circular_x:
        dx %= 160
        dx = min(dx, 160 - dx) / 80
    else:
        dx /= 160
    dy = abs((ay + ah / 2) - (by + bh / 2)) / 210
    distance = min(1.0, (dx + dy) / 2)
    if a["kind"] == "floe":
        different_color = a["attributes"].get("white") != b["attributes"].get("white")
        distance = (distance + float(different_color)) / 2
    return distance


def set_distance(left, right, circular_x=False):
    """Padded Hungarian matching: absent object costs 1, two empty sets cost 0.

    Objects match by kind and geometry, not by RAM slot/track ID. Normalize by
    max cardinality so a large set does not dominate the other semantic groups.
    """
    n = max(len(left), len(right))
    if n == 0:
        return 0.0
    if not left or not right:
        return 1.
    costs = np.ones((n, n), dtype=np.float64)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            costs[i, j] = object_distance(a, b, circular_x)
    return assignment_cost(costs) / n


def semantic_distance(game, anchor, neighbor):
    """Return equal-group-weight distance in [0,1], similarity=1-distance.

    Invalid/unknown RAM interpretations raise instead of being scored as absent.
    """
    if anchor["warnings"] or neighbor["warnings"]:
        raise ValueError("RAM mapping warning: " + "; ".join(anchor["warnings"] + neighbor["warnings"]))
    a, b = anchor["scalars"], neighbor["scalars"]
    select = lambda state, kinds: [o for o in state["objects"] if o["kind"] in kinds]
    group = lambda kinds, wrap=False: set_distance(select(anchor, kinds), select(neighbor, kinds), wrap)
    if game == "seaquest":
        groups = dict(player=group({"player"}),
                      enemies=group({"shark", "enemy_submarine", "enemy_missile"}),
                      divers=group({"diver"}),
                      collected_divers=abs(a["collected_divers"] - b["collected_divers"]) / 6,
                      oxygen=min(1.0, abs(a["oxygen"] - b["oxygen"]) / 64))
    elif game == "frostbite":
        if a["igloo_stage"] is None or b["igloo_stage"] is None:
            raise ValueError("Unknown igloo stage")
        groups = dict(player=group({"player"}),
                      enemies=group({"bird", "crab", "clam", "fish", "bear"}),
                      floes=group({"floe"}, True),
                      igloo_stage=abs(a["igloo_stage"] - b["igloo_stage"]) / 16)
    elif game == "pong":
        groups = {kind: group({kind}) for kind in GROUPS[game]}
    else:
        raise ValueError(f"Unsupported evaluation game: {game}")
    distance = sum(groups.values()) / len(groups)
    if not math.isfinite(distance) or not 0 <= distance <= 1:
        raise ValueError(f"Invalid semantic distance {distance}")
    return dict(distance=distance, similarity=1 - distance,
                **{f"distance_{key}": value for key, value in groups.items()})


def ram_distance(game, anchor_ram, neighbor_ram):
    return semantic_distance(game, extract_state(game, anchor_ram), extract_state(game, neighbor_ram))


def summarize_queries(queries, seed=0, bootstrap_samples=5000):
    """Query-macro means; paired bootstrap over seeds, or episodes for one seed."""
    paired = [q for q in queries if q.get("retrieval_distance") is not None
              and q.get("random_distance") is not None]
    summary = dict(query_count=len(queries), paired_query_count=len(paired),
                   queries_with_neighbors=sum(q["neighbor_count"] > 0 for q in queries),
                   queries_with_valid_pairs=len(paired),
                   mean_neighbors=float(np.mean([q["neighbor_count"] for q in queries])) if queries else 0,
                   coverage=sum(q["neighbor_count"] > 0 for q in queries) / len(queries) if queries else 0,
                   valid_retrieval_pair_count=sum(q.get("valid_retrieval_pairs", 0) for q in queries),
                   valid_random_pair_count=sum(q.get("valid_random_pairs", 0) for q in queries),
                   aggregation="equal weight per query; paired CI resamples rollout seeds, or anchor episodes with one seed")
    if not paired:
        summary["status"] = "no_valid_paired_queries"
        return summary
    retrieval = np.array([q["retrieval_distance"] for q in paired])
    random = np.array([q["random_distance"] for q in paired])
    delta = random - retrieval
    summary.update(status="ok", retrieval_mean_distance=float(retrieval.mean()),
                   random_mean_distance=float(random.mean()),
                   retrieval_mean_similarity=float(1 - retrieval.mean()),
                   random_mean_similarity=float(1 - random.mean()),
                   paired_distance_reduction=float(delta.mean()),
                   relative_distance_reduction=float(1 - retrieval.mean() / random.mean()) if random.mean() > 0 else None)
    episode_keys = [(q.get("evaluation_seed", 0), q["anchor_episode"]) for q in paired]
    rollout_seeds = sorted(set(key[0] for key in episode_keys))
    keys = [(key[0],) for key in episode_keys] if len(rollout_seeds) >= 2 else episode_keys
    clusters = {key: i for i, key in enumerate(sorted(set(keys)))}
    cluster_ids = np.asarray([clusters[key] for key in keys])
    summary["anchor_episode_clusters"] = len(set(episode_keys))
    summary["evaluation_seeds"] = rollout_seeds
    summary["bootstrap_unit"] = "rollout_seed" if len(rollout_seeds) >= 2 else "anchor_episode"
    summary["bootstrap_clusters"] = len(clusters)
    summary["bootstrap_samples"] = bootstrap_samples
    summary["paired_distance_reduction_ci95"] = None
    summary["retrieval_mean_similarity_ci95"] = None
    summary["random_mean_similarity_ci95"] = None
    if len(clusters) >= 2 and bootstrap_samples:
        sums = np.stack([np.bincount(cluster_ids, weights=values, minlength=len(clusters))
                         for values in (retrieval, random, delta)], axis=1)
        counts = np.bincount(cluster_ids, minlength=len(clusters))
        rng, boot = np.random.default_rng(seed), []
        for start in range(0, bootstrap_samples, 128):
            indexes = rng.integers(0, len(clusters), (min(128, bootstrap_samples - start), len(clusters)))
            boot.append(sums[indexes].sum(1) / counts[indexes].sum(1)[:, None])
        intervals = np.quantile(np.concatenate(boot), [.025, .975], axis=0)
        summary["paired_distance_reduction_ci95"] = intervals[:, 2].tolist()
        summary["retrieval_mean_distance_ci95"] = intervals[:, 0].tolist()
        summary["random_mean_distance_ci95"] = intervals[:, 1].tolist()
        summary["retrieval_mean_similarity_ci95"] = (1 - intervals[::-1, 0]).tolist()
        summary["random_mean_similarity_ci95"] = (1 - intervals[::-1, 1]).tolist()
    else:
        summary["uncertainty_note"] = "At least two clusters are needed; no IID-pair interval is substituted. Use multiple rollout seeds for run-level uncertainty."
    return summary
