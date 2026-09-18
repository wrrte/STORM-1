import wandb
import pandas as pd
import json

def check_run(run_name_query):
    print(f"Checking W&B for run name containing: {run_name_query}")
    api = wandb.Api()
    
    # Fetch runs
    entity = "choemj-kaist"
    print(f"Querying project: {entity}/STORM")
    runs = api.runs(f"{entity}/STORM")
    target_run = None
    for r in runs:
        if run_name_query in r.name:
            target_run = r
            break
            
    if target_run is None:
        print(f"Run '{run_name_query}' not found in {entity}/STORM.")
        # Try default entity as fallback
        entity = api.default_entity
        print(f"Trying default entity: {entity}/STORM")
        runs = api.runs(f"{entity}/STORM")
        for r in runs:
            if run_name_query in r.name:
                target_run = r
                break
                
        if target_run is None:
            print(f"Run '{run_name_query}' not found in W&B at all.")
            return

    print(f"\n[Run Found]")
    print(f"Project: {target_run.project}")
    print(f"Entity: {target_run.entity}")
    print(f"Name: {target_run.name}")
    print(f"ID: {target_run.id}")
    print(f"State: {target_run.state}")
    
    # Check config
    config = target_run.config
    print("\n[Configuration]")
    
    # Retrieve nested config safely
    try:
        ret_enable = config.get('JointTrainAgent', {}).get('Retrieval', {}).get('enable', 'Not found')
        target = config.get('JointTrainAgent', {}).get('Retrieval', {}).get('target', 'Not found')
        warmup_steps = config.get('Agent', {}).get('warmup_steps', 'Not found')
        bsr = config.get('JointTrainAgent', {}).get('Retrieval', {}).get('batch_size_reduction', 'Not found')
        z_score = config.get('JointTrainAgent', {}).get('Retrieval', {}).get('z_score_threshold', 'Not found')
        hash_bits = config.get('JointTrainAgent', {}).get('Retrieval', {}).get('hash_bits', 'Not found')
    except AttributeError:
        # If it's a flat structure
        ret_enable = config.get('JointTrainAgent.Retrieval.enable', 'Not found')
        target = config.get('JointTrainAgent.Retrieval.target', 'Not found')
        warmup_steps = config.get('Agent.warmup_steps', 'Not found')
        bsr = config.get('JointTrainAgent.Retrieval.batch_size_reduction', 'Not found')
        z_score = config.get('JointTrainAgent.Retrieval.z_score_threshold', 'Not found')
        hash_bits = config.get('JointTrainAgent.Retrieval.hash_bits', 'Not found')
        
    seed = config.get('Seed', 'Not found')
    
    print(f"- Retrieval Enable: {ret_enable}")
    print(f"- Target: {target}")
    print(f"- Warmup Steps: {warmup_steps}")
    print(f"- Batch Size Reduction: {bsr}")
    print(f"- Z Score Threshold: {z_score}")
    print(f"- Hash Bits: {hash_bits}")
    print(f"- Seed: {seed}")
    
    # Check summary metrics
    summary = target_run.summary._json_dict
    env_steps = summary.get('env_step', 'Not found')
    eval_return = summary.get('eval/episode_avg_return', 'Not found')
    print("\n[Summary Metrics]")
    print(f"- Env Step: {env_steps}")
    print(f"- Eval Return (avg): {eval_return}")
    
    if eval_return == 'Not found':
        print("\n-> eval/episode_return is missing from summary! Checking history...")
        history = target_run.scan_history(keys=['env_step', 'eval/episode_return'])
        returns = [row.get('eval/episode_return') for row in history if row.get('eval/episode_return') is not None]
        if returns:
            print(f"-> Found {len(returns)} eval returns in history. Last recorded: {returns[-1]}")
        else:
            print("-> CRITICAL: No eval returns found in history! This means the run didn't log any scores.")
            
    # Check classify logic constraints
    print("\n[classify_wandb_runs.py Check]")
    if eval_return == 'Not found' and not returns:
        print("-> Run will fail in classify_wandb_runs.py because eval_return is missing.")
    elif target_run.state != "finished":
        print(f"-> Warning: Run state is '{target_run.state}'. Depending on how far it ran, it might not have final scores.")
    else:
        print("-> classify_wandb_runs.py criteria seems OK (Assuming score exists).")
        
    # Check convert_csv_to_excel.py constraints
    print("\n[convert_csv_to_excel.py Check]")
    created_at = target_run.created_at
    run_date = pd.to_datetime(created_at, utc=True)
    ema_applied_date = pd.to_datetime("2026-08-24T11:05:43Z", utc=True)
    
    print(f"- Run Created At: {run_date}")
    print(f"- EMA Applied Date: {ema_applied_date}")
    
    if str(ret_enable).strip().lower() in ['true', '1', 't']:
        if run_date < ema_applied_date:
            print("-> ❌ FILTERED OUT: Run is using Retrieval (True) but was created BEFORE ema_applied_date. It will be skipped.")
        else:
            print("-> ✅ PASSED date check.")
    else:
        print("-> ✅ PASSED date check (Retrieval 미사용 bypasses date check).")

if __name__ == "__main__":
    check_run("Qbert_moln5tny_10_X")
