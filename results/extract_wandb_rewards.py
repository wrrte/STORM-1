import os
import json
import wandb
import subprocess
from tqdm import tqdm

def get_logic_for_commit(commit_hash):
    if not commit_hash:
        return "Unknown (No commit hash)"
    
    try:
        result = subprocess.run(
            ["git", "show", f"{commit_hash}:train.py"],
            capture_output=True, text=True, check=True
        )
        code = result.stdout
        
        if "random_batch_size = max(0, imagine_batch_size - retrieved_count)" in code and "random_batch_size = max(0, imagine_batch_size - num_valid_anchors)" in code:
            return "num_valid_anchors"
        elif "random_batch_size = imagine_batch_size - num_valid_anchors" in code or "random_batch_size = max(0, imagine_batch_size - num_valid_anchors)" in code:
            return "num_valid_anchors"
        elif "random_batch_size = max(0, imagine_batch_size - retrieved_count)" in code:
            return "retrieved_count (with max)"
        elif "random_batch_size = imagine_batch_size - retrieved_count" in code:
            return "retrieved_count (without max)"
        elif "random_batch_size = imagine_batch_size" in code:
            return "imagine_batch_size (No retrieval deduction)"
        else:
            return "Unknown (Logic not found in code)"
            
    except subprocess.CalledProcessError:
        return "Unknown (Git error or train.py missing)"

def get_config_val(config, key_path):
    if key_path in config:
        return config[key_path]
    keys = key_path.split('.')
    val = config
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return None
    return val

def main():
    api_key_path = os.path.join(os.path.dirname(__file__), '.wandb_api_key')
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as f:
            api_key = f.read().strip()
            wandb.login(key=api_key)
            print("✅ Successfully logged in using .wandb_api_key")
    else:
        print("⚠️ .wandb_api_key not found. Attempting default credentials.")

    api = wandb.Api()
    try:
        entity = api.default_entity
        wandb_path = f"{entity}/STORM"
        print(f"Fetching runs from '{wandb_path}'...")
        runs = api.runs(wandb_path)
    except Exception as e:
        print(f"Failed to fetch runs: {e}")
        return

    extracted_data = []

    for run in tqdm(runs, desc="Processing runs"):
        # 1. Parsing Meta Data
        run_name = run.name
        parts = run_name.split('_')
        game = parts[0]
        
        commit_hash = run.commit
        if not commit_hash and 'commit' in run.config:
            commit_hash = run.config['commit']
            
        logic_type = get_logic_for_commit(commit_hash)
        
        eval_return = run.summary.get('eval/episode_avg_return', 'N/A')
        
        ret_enable = get_config_val(run.config, 'JointTrainAgent.Retrieval.enable')
        ret_enable = bool(ret_enable) if ret_enable is not None else False
        
        warmup_steps = 'N/A'
        if ret_enable:
            w_steps = get_config_val(run.config, 'JointTrainAgent.Retrieval.warmup_steps')
            warmup_steps = w_steps if w_steps is not None else 'N/A'
            
        seed = get_config_val(run.config, 'seed')
        if seed is None or seed == 0:
            if len(parts) >= 4 and parts[-1] in ['O', 'X']:
                try:
                    seed = int(parts[-2])
                except ValueError:
                    pass
            elif len(parts) >= 3 and parts[2].isdigit():
                seed = int(parts[2])
            elif len(parts) == 4 and parts[3].isdigit():
                seed = int(parts[3])
                
        # Derive final config string matching excel
        if not ret_enable:
            config = 'retrieval false'
        else:
            logic_str = str(logic_type).strip()
            if 'without max' in logic_str:
                config = 'retrieved_count (without max)'
            elif 'with max' in logic_str:
                config = 'retrieved_count (with max)'
            elif 'num_valid_anchors' in logic_str:
                config = 'num_valid_anchors'
            elif 'imagine_batch_size' in logic_str:
                config = 'imagine_batch_size (No retrieval deduction)'
            else:
                config = logic_str

        # Ensure seed is an integer for clean grouping
        try:
            if isinstance(seed, str) and seed.endswith('.0'):
                seed = int(float(seed))
            else:
                seed = int(seed)
        except (ValueError, TypeError):
            pass # Keep as is if it can't be parsed

        run_data = {
            "run_name": run.name,
            "run_id": run.id,
            "game": game,
            "seed": seed,
            "retrieval_enable": ret_enable,
            "config": config,
            "eval_return": eval_return,
            "warmup_steps": warmup_steps,
            "reward_history": {"step": [], "reward": []}
        }

        # 2. Extract Reward History ONLY for baseline (retrieval false) runs
        if not ret_enable:
            # Find the reward key in summary
            reward_key = None
            for key in run.summary.keys():
                if key.startswith('sample/') and key.endswith('_reward'):
                    reward_key = key
                    break
            
            if reward_key:
                print(f"  -> Found reward graph: {reward_key}")
                try:
                    # _step을 명시적으로 요구하면 스키마 에러가 나는 경우가 있어 안전하게 history() 사용
                    # 만약 데이터가 너무 많아 잘린다면 scan_history(keys=[reward_key])로 변경
                    history = run.history(keys=[reward_key], samples=1000000, pandas=False)
                    steps = []
                    rewards = []
                    for row in history:
                        if reward_key in row and row[reward_key] is not None:
                            # run.history()는 기본적으로 '_step'을 포함하여 반환합니다
                            step_val = row.get('_step', len(steps))
                            steps.append(step_val)
                            rewards.append(row[reward_key])
                    
                    run_data["reward_history"]["step"] = steps
                    run_data["reward_history"]["reward"] = rewards
                    print(f"  -> Extracted {len(steps)} data points.")
                except Exception as e:
                    print(f"  -> Warning: Failed to fetch history for run {run.name}: {e}")
            else:
                print(f"  -> Warning: Could not find any key matching 'sample/*_reward' for run {run.name}")

        extracted_data.append(run_data)

    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), 'wandb_extracted_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Extraction complete! Data saved to {output_path}")

if __name__ == "__main__":
    main()
