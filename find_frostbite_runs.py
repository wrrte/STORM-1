import wandb

def find_frostbite_710_runs(project_name="STORM"):
    # wandb API 객체 초기화
    api = wandb.Api()
    
    try:
        # 특정 프로젝트의 모든 run을 가져옵니다. 
        # (만약 entity가 필요하다면 "entity_name/STORM" 형태로 변경해야 할 수 있습니다.)
        runs = api.runs(project_name)
    except Exception as e:
        print(f"프로젝트 '{project_name}'에서 run을 가져오는데 실패했습니다: {e}")
        return

    matched_runs = []
    
    print("조건에 맞는 Run을 검색 중입니다...\n")
    for run in runs:
        is_match = False
        
        # 1. Run Name을 통한 확인
        # utils.py의 Logger에 따르면, run name은 f"{pure_env_name}_{run_id}_{seed}{suffix}" 형태로 저장됩니다.
        if run.name and run.name.startswith("Frostbite_"):
            parts = run.name.split('_')
            # parts 예시: ['Frostbite', 'runid', '710', 'O']
            if "710" in parts:
                is_match = True

        # 2. Config를 통한 확인 (만약 config에 정보가 저장되어 있는 경우)
        if not is_match:
            config = run.config
            env_name = config.get("env_name", "")
            seed = config.get("seed", None)
            
            # env_name에 Frostbite가 포함되어 있고 seed가 710인 경우
            if env_name and "Frostbite" in str(env_name) and str(seed) == "710":
                is_match = True

        if is_match:
            matched_runs.append(run)
            
    if not matched_runs:
        print("Frostbite 환경에서 시드(seed)가 710인 run을 찾지 못했습니다.")
    else:
        print(f"총 {len(matched_runs)}개의 매칭되는 Run을 찾았습니다:")
        for idx, run in enumerate(matched_runs, 1):
            print(f"[{idx}] Run ID : {run.id}")
            print(f"    Name   : {run.name}")
            print(f"    State  : {run.state}")
            print(f"    URL    : {run.url}")
            print("-" * 50)

if __name__ == "__main__":
    find_frostbite_710_runs("STORM")
