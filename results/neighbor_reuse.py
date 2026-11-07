"""Read the audited historical-seed neighbor comparison and its control jobs."""

import json
from pathlib import Path

import pandas as pd


STUDY = 'neighbor_reuse_20261107'


def load_reuse_plan(queue_dir):
    path = Path(queue_dir)/'results'/STUDY/'plan.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def reuse_control_names(plan):
    return {job['final_run_name'] for job in plan['training_jobs']} if plan else set()


def reuse_study_table(frame, queue_dir, queued_rows, plan):
    """Keep historical random scores fixed and match each to its FLASH seed."""
    records = frame.to_dict('records')
    pending = {row.get('Local Run Name') for row in queued_rows}
    jobs = {(job['game'], job['seed']): job for job in plan['training_jobs']}
    output = []
    for pair in plan['comparisons']:
        game, seed = pair['game'], pair['seed']
        random = pair['random_result']
        flash = pair['flash_result']
        job = jobs.get((game, seed))
        state = 'FINISHED'
        if flash is None:
            matches = [row for row in records if row.get('Local Run Name') == job['final_run_name']]
            selected = max(matches, key=lambda row: str(row.get('Created At', '')), default={})
            score = pd.to_numeric(selected.get('Eval Return'), errors='coerce')
            if str(selected.get('State', '')).lower() == 'finished' and pd.notna(score):
                flash = {'score': float(score), 'run_id': selected.get('Run ID', ''),
                         'local_run_name': job['final_run_name'], 'source': 'new matched FLASH control'}
            else:
                state = ('QUEUED FLASH' if job['final_run_name'] in pending else
                         'RUNNING FLASH' if selected or (Path(queue_dir)/'runs'/job['final_run_name']).exists()
                         else 'AWAITING FLASH')
        flash_score = flash['score'] if flash else None
        output.append({
            'Study': STUDY, 'Game': game, 'Seed': seed, 'Status': state,
            'Full FLASH': flash_score, 'Random Neighbor': random['score'],
            'Random minus FLASH': random['score']-flash_score if flash else None,
            'Random Source': 'historical completed run', 'Random W&B Run ID': random['run_id'],
            'Full FLASH Source': flash.get('source', 'historical completed run') if flash else '',
            'Full FLASH W&B Run ID': flash['run_id'] if flash else '',
            'Full FLASH Local Run Name': flash.get('local_run_name', '') if flash else job['final_run_name'],
            'Local Run Name': random['local_run_name'],
            'Warmup Steps': plan['warmup_steps'], 'Anchor Offset': plan['anchor_offset'],
            'Eval Checkpoint Step': plan['evaluation']['checkpoint_step'],
            'Final Evaluation': job['final_evaluation_path'] if job else '',
        })
    return pd.DataFrame(output)
