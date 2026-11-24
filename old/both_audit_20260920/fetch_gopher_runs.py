"""Read only the six W&B runs supplied for the Both investigation."""

import json
from pathlib import Path
import socket

import wandb

OUTPUT = Path(__file__).resolve().parent
STORM = OUTPUT.parents[1]
RUN_IDS = ("3d8lc1uk", "tepmvj6g", "w30fmuzh", "ori7lpmm", "hrca6yyg", "uz4318p4",
           "gsvmbdpy", "ic64hxjj", "9uwipoit", "gu3gyrb0", "55cqyaw3")


def main():
    socket.getaddrinfo("api.wandb.ai", 443)
    key_file = next((p for p in (STORM / ".wandb_api_key", STORM / "results/.wandb_api_key") if p.exists()), None)
    api = wandb.Api(api_key=key_file.read_text().strip() if key_file else None, timeout=20)
    parents = list(api.runs("choemj-kaist/STORM", filters={
        "display_name": {"$regex": "^Gopher_.*_(6000|6010|9999)$"},
        "createdAt": {"$gte": "2026-09-17T00:00:00Z"},
    }))
    print("Warmup parent candidates:", [(r.id, r.name) for r in parents], flush=True)
    runs = []
    missing = []
    for run_id in dict.fromkeys([*RUN_IDS[:6], *(r.id for r in parents), *RUN_IDS[6:]]):
        cached = OUTPUT / run_id / "run_data.json"
        if cached.exists():
            entry = json.loads(cached.read_text())
            if entry.get("first_updates") and "WorldModel/total_loss" in entry["first_updates"][0] and len(entry.get("episode_rewards", [])) < 1000:
                runs.append(entry)
                continue
        try:
            run = api.run(f"choemj-kaist/STORM/{run_id}")
        except wandb.errors.CommError:
            missing.append(run_id)
            print("Unavailable run:", run_id, flush=True)
            continue
        target = OUTPUT / run_id
        target.mkdir(exist_ok=True)
        summary = dict(run.summary)
        summary = {key: value for key, value in summary.items()
                   if key.startswith(("ActorCritic/", "WorldModel/", "Retrieval/", "sample/", "eval/", "replay_buffer/"))
                   and isinstance(value, (str, float, int, bool))}
        metadata_file = run.file("wandb-metadata.json").download(root=str(target), replace=True)
        metadata = json.loads(Path(metadata_file.name).read_text())
        entry = {
            "id": run.id, "name": run.name, "state": run.state, "created_at": run.created_at,
            "config": {key: value for key, value in run.config.items() if not key.startswith("_")},
            "summary": summary,
            "metadata": {key: metadata.get(key) for key in ("args", "gpu", "python", "executable", "cudaVersion", "git", "host")},
        }
        history = run.history(samples=1500, pandas=False)
        entry["history"] = [{key: value for key, value in row.items()
                             if (key.startswith(("ActorCritic/", "WorldModel/", "Retrieval/", "sample/", "eval/", "replay_buffer/"))
                                 or key in ("_step", "_timestamp", "_runtime"))
                             and isinstance(value, (str, float, int, bool))} for row in history]
        # The installed SDK's scan_history returned unrelated columns and then
        # failed on a missing _step schema. Use its GraphQL history path instead.
        entry["episode_rewards"] = run.history(keys=["sample/ALE/Gopher-v5_reward"], samples=100000, pandas=False)
        query = """query RunSampledHistory($project: String!, $entity: String!, $name: String!, $specs: [JSONString!]!) {
            project(name: $project, entityName: $entity) { run(name: $name) { sampledHistory(specs: $specs) } }
        }"""
        spec = {"keys": ["_step", "WorldModel/total_loss", "ActorCritic/norm_ratio", "ActorCritic/entropy_loss"],
                "minStep": 0, "maxStep": 20, "samples": 20}
        entry["first_updates"] = run._exec(query, specs=[json.dumps(spec)])["project"]["run"]["sampledHistory"][0]
        (target / "run_data.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2))
        runs.append(entry)
        (OUTPUT / "gopher_runs.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2))
        print(json.dumps({"name": entry["name"], "state": entry["state"], "created_at": entry["created_at"],
                          "gpu": entry["metadata"]["gpu"], "args": entry["metadata"]["args"],
                          "eval_return": summary.get("eval/episode_avg_return"),
                          "history_rows": len(history)}, ensure_ascii=False), flush=True)
    (OUTPUT / "gopher_runs.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2))
    (OUTPUT / "unavailable_runs.json").write_text(json.dumps(missing))


if __name__ == "__main__":
    main()
