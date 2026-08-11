import sys
import os

# Remove current directory from sys.path to prevent importing local 'wandb' folder
if '' in sys.path: sys.path.remove('')
if os.getcwd() in sys.path: sys.path.remove(os.getcwd())

import wandb

def main():
    api = wandb.Api()
    
    project_name = "STORM"
    try:
        entity = api.default_entity
        runs = api.runs(f"{entity}/{project_name}")
    except Exception:
        runs = api.runs(project_name)

    valid_runs = []
    
    print("Fetching runs...")
    for run in runs:
        # Check if the run is related to Frostbite
        if "Frostbite" not in run.name and "Frostbite" not in str(run.config.get("env_name", "")):
            continue
            
        config = run.config
        
        # Check Retrieval conditions
        has_retrieval = False
        enable_val = None
        warmup_val = None
        
        # 1. Try to find Retrieval in nested dict
        def find_retrieval(c):
            if isinstance(c, dict):
                if "Retrieval" in c:
                    return c["Retrieval"]
                for k, v in c.items():
                    res = find_retrieval(v)
                    if res is not None: return res
            return None
            
        retrieval_dict = find_retrieval(config)
        
        if retrieval_dict is None:
            # 2. Try flattened keys
            flat_retrieval = {k: v for k, v in config.items() if "Retrieval" in k}
            if flat_retrieval:
                has_retrieval = True
                for k, v in flat_retrieval.items():
                    if k.endswith(".enable") or k == "Retrieval.enable" or "enable" in k.split("."):
                        enable_val = v
                    if k.endswith(".warmup") or k == "Retrieval.warmup" or "warmup" in k.split("."):
                        warmup_val = v
        else:
            has_retrieval = True
            if isinstance(retrieval_dict, dict):
                enable_val = retrieval_dict.get("enable")
                warmup_val = retrieval_dict.get("warmup")
                
        # Evaluate condition
        is_valid_retrieval = False
        if not has_retrieval:
            is_valid_retrieval = True
        elif enable_val in [False, "False", "false", 0, None]:
            is_valid_retrieval = True
        elif warmup_val is not None:
            try:
                if int(warmup_val) >= 14000:
                    is_valid_retrieval = True
            except ValueError:
                pass
                
        if not is_valid_retrieval:
            continue
            
        print(f"Checking run: {run.name} (ID: {run.id})")
        
        try:
            # Use run.history with pandas=False to avoid pandas dependency and '_step' errors
            history = run.history(keys=["sample/ALE/Frostbite-v5_reward"], pandas=False)
            max_reward = -float('inf')
            
            for row in history:
                reward = row.get("sample/ALE/Frostbite-v5_reward")
                if reward is not None and reward > max_reward:
                    max_reward = reward
                    
            if max_reward >= 600:
                print(f"  -> [MATCH] Max Reward: {max_reward}")
                valid_runs.append((run.name, run.url, max_reward))
            else:
                if max_reward == -float('inf'):
                    print("  -> No reward data found (or max was -inf).")
                else:
                    print(f"  -> Max Reward was {max_reward} (less than 600)")
                    
        except Exception as e:
            print(f"  -> Error checking history: {e}")

    print("\n" + "="*50)
    print("MATCHING RUNS (Reward >= 600)")
    print("="*50)
    if not valid_runs:
        print("No runs found meeting the criteria.")
    else:
        for name, url, reward in valid_runs:
            print(f"Run: {name} | Max Reward: {reward} | URL: {url}")

if __name__ == "__main__":
    main()
