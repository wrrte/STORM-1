"""Show the training queues for the GPU models installed on this machine."""

import argparse
from collections import Counter
from itertools import groupby
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

from queue_train import ROOT, gpu_type
from storm_warmup_sweep import experiment_names, hash_target_variants
from warmup_catalog import choose_warmup, request_config, source_stamp


def detect_models(driver_dir=Path("/proc/driver/nvidia/gpus")):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        names = []
    if not names:
        for path in driver_dir.glob("*/information"):
            for line in path.read_text().splitlines():
                key, _, value = line.partition(":")
                if key.strip() == "Model" and value.strip():
                    names.append(value.strip())
                    break
    if not names:
        raise ValueError("이 서버의 NVIDIA GPU 모델을 감지하지 못했습니다.")
    return sorted({gpu_type(name) for name in names})


def read_jobs(database, models, show_all):
    if not database.exists():
        return []
    placeholders = ",".join("?" for _ in models)
    # Open read-only so checking status cannot recover, claim, or update jobs.
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(f"""
            SELECT id, gpu_type, status, priority, worker, payload FROM jobs
            WHERE gpu_type IN ({placeholders}) AND (? OR status IN ('running', 'pending'))
            ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END,
                     priority DESC, seq
        """, (*models, show_all)).fetchall()
    return [dict(json.loads(payload), id=identity, gpu_type=model, status=status,
                 priority=priority, worker=json.loads(worker) if worker else None)
            for identity, model, status, priority, worker, payload in rows]


def read_inventory(root, models):
    try:
        data = json.loads((root / "training_queue/warmup_inventory.json").read_text())
    except FileNotFoundError:
        return None
    reusable = []
    for record in data["available"]:
        if record["gpu_type"] not in models:
            continue
        source = root / record["source"]
        if (all((source / name).is_file() for name in ("training_state.pt", "replay_buffer.npz"))
                and record.get("stamp") == [list(item) for item in source_stamp(source)]):
            reusable.append(record)
    return {"available": reusable}


def stage_label(stage):
    if stage == "baseline":
        return "baseline 평가"
    if re.search(r"_h\d+_t\d+$", stage):
        return "해시비트-타겟 스윕"
    return stage


def describe_job(job, inventory):
    stages = list(job.get("plan") or {})
    source = job.get("warmup_source") or job.get("resume_warmup")
    if not stages:
        config = request_config(job).JointTrainAgent.Retrieval
        stages = [name + suffix for name, suffix, _ in hash_target_variants(
            experiment_names(config.enable), config.get("hash_target_pairs", []))]
        if not source and inventory is not None:
            record, _ = choose_warmup(job, inventory)
            if record:
                source = record["source"]

    if source:
        collector = f"{job['name']}_collector_Shared"
        warmup = "신규 warmup 수집" if Path(source).parent.name == collector else "기존 warmup 재사용"
    elif inventory is not None and job["status"] in ("pending", "running"):
        warmup = "신규 warmup 수집 예정" if job["status"] == "pending" else "warmup 준비·수집 중"
    else:
        warmup = "warmup 미확정"

    steps = [warmup]
    for label, group in groupby(map(stage_label, stages)):
        count = len(list(group))
        steps.append(f"{label} ({count}분기)" if count > 1 else label)
    completed = set(job.get("completed_branches", ()))
    progress = f"{sum(stage in completed for stage in stages)}/{len(stages)}"
    if job["status"] == "running":
        next_stage = next((stage for stage in stages if stage not in completed), None)
        if next_stage is not None:
            progress += f" · {next_stage if job.get('plan') else 'warmup 준비'}"
    return " → ".join(steps), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Include completed, failed and paused jobs")
    args = parser.parse_args()
    try:
        models = detect_models()
        jobs = read_jobs(ROOT / "training_queue/queue.sqlite3", models, args.all)
        inventory = read_inventory(ROOT, models) if any(not job.get("plan") for job in jobs) else None
        for model in models:
            selected = [job for job in jobs if job["gpu_type"] == model]
            counts = Counter(job["status"] for job in selected)
            summary = f"실행 중 {counts['running']}개 · 대기 {counts['pending']}개"
            if args.all:
                summary += f" · 완료 {counts['completed']}개 · 실패 {counts['failed']}개 · 일시정지 {counts['paused']}개"
            print(f"\n[{model}] {summary}")
            for job in selected:
                description, progress = describe_job(job, inventory)
                worker = job["worker"] or {}
                device = f"GPU {worker['index']} · " if "index" in worker else ""
                print(f"  {job['status']:<9} p={job['priority']}  {job['game']}/{job['seed']}  "
                      f"{device}{description}  [{progress}]  {job['id']}")
    except (ImportError, KeyError, OSError, ValueError, sqlite3.Error) as error:
        print(f"큐 조회 오류 · {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
