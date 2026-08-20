import sys
import os
sys.path = [p for p in sys.path if p and p != os.path.dirname(os.path.abspath(__file__))]
import wandb
import subprocess
import os
import csv
from collections import Counter

def get_logic_for_commit(commit_hash):
    """
    주어진 커밋 해시에서 train.py 파일을 읽어와 
    random_batch_size 설정 로직을 파악합니다.
    """
    if not commit_hash:
        return "Unknown (No commit hash)"
    
    try:
        # git show <commit>:train.py 명령어로 해당 커밋 시점의 코드 내용 추출
        result = subprocess.run(
            ["git", "show", f"{commit_hash}:train.py"],
            capture_output=True, text=True, check=True
        )
        code = result.stdout
        
        # 코드 내용에 따른 분류
        # 최신 코드의 경우 num_valid_anchors와 retrieved_count 조건이 모두 포함될 수 있으므로, 두 개가 다 있을 땐 num_valid_anchors로 분류
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
        # 파일이 존재하지 않거나 커밋을 찾을 수 없는 경우
        return "Unknown (Git error or train.py missing)"

def main():
    # 1. 자동 로그인 처리
    api_key_path = os.path.join(os.path.dirname(__file__), '.wandb_api_key')
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as f:
            api_key = f.read().strip()
            wandb.login(key=api_key)
            print("✅ .wandb_api_key를 사용하여 성공적으로 로그인했습니다.")
    else:
        print("⚠️ .wandb_api_key 파일을 찾을 수 없습니다. 기존 설정된 자격 증명을 시도합니다.")

    # 2. WandB API 객체 생성 및 엔티티 자동 추출
    api = wandb.Api()
    try:
        # 로그인된 계정의 기본 엔티티 이름 가져오기
        entity = api.default_entity
        wandb_path = f"{entity}/STORM"
        print(f"'{wandb_path}' 프로젝트의 run들을 분석합니다...")
        runs = api.runs(wandb_path)
    except Exception as e:
        print(f"WandB API 호출 실패: {e}")
        return

    results = []
    logic_counts = Counter()
    
    print(f"총 {len(runs)}개의 run을 확인했습니다. 분류를 시작합니다...\n")
    
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

    for run in runs:
        # WandB는 기본적으로 github 연동이나 git 추적 시 commit 정보를 남깁니다.
        commit_hash = run.commit
        
        # run.commit이 없는 경우 config나 summary 등 다른 곳에 수동 기록했는지 확인
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
            # 설정 파일에 제대로 저장되지 않은 경우 런 이름에서 유추
            # 이름 형식: {env}_{id}_{seed}_{O/X} 혹은 {env}_{id}_{O/X}
            parts = run.name.split('_')
            if len(parts) >= 4 and parts[-1] in ['O', 'X']:
                try:
                    seed = int(parts[-2])
                except ValueError:
                    pass
        seed = seed if seed is not None else 'N/A'
        
        results.append({
            "Run Name": run.name,
            "Run ID": run.id,
            "State": run.state,
            "Commit": commit_hash[:7] if commit_hash else "None",
            "Logic": logic_type,
            "Eval Return": eval_return,
            "Retrieval Enable": ret_enable,
            "Warmup Steps": warmup_steps,
            "Seed": seed
        })
        logic_counts[logic_type] += 1
        print(f"Run: {run.name:20} | Commit: {str(commit_hash)[:7]:7} | Return: {str(eval_return)[:8]:8} | Logic: {logic_type} | Ret: {ret_enable} | Warmup: {warmup_steps} | Seed: {seed}")

    # 요약 및 저장
    print("\n" + "="*50)
    print("분류 요약:")
    for logic, count in logic_counts.most_common():
        print(f"  {logic}: {count} runs")
    print("="*50)
    
    output_csv = "wandb_runs_classification.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Run Name", "Run ID", "State", "Commit", "Logic", "Eval Return", "Retrieval Enable", "Warmup Steps", "Seed"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"\n상세 결과가 '{output_csv}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()
