import wandb

def find_target_runs(project_name="STORM"):
    api = wandb.Api()
    
    try:
        # 본인의 wandb entity명이 필요한 경우 "entity_name/STORM" 형태로 기입하세요.
        runs = api.runs(project_name)
    except Exception as e:
        print(f"프로젝트 '{project_name}'에서 run을 가져오는데 실패했습니다: {e}")
        return

    matched_runs = []
    
    print("조건에 맞는 Run을 검색 중입니다...\n")
    for run in runs:
        # 1. Frostbite 환경에서 학습되었는지 확인
        is_frostbite = "Frostbite" in run.name or "Frostbite" in str(run.config.get("env_name", ""))
        if not is_frostbite:
            continue
            
        # 1-1. seed가 3710인지 확인
        is_seed_match = False
        if "3710" in run.name.split('_'):
            is_seed_match = True
        elif str(run.config.get("seed", "")) == "3710":
            is_seed_match = True
            
        if not is_seed_match:
            continue

        # 2. train.py를 통해 학습되었는지 확인 
        # (wandb config에 저장된 실행 파일명(program)을 확인합니다. 상황에 따라 기록되지 않을 수 있으니 주석 해제하여 사용하세요)
        # program_name = run.config.get("program", "")
        # if program_name and "train.py" not in program_name:
        #     continue

        # 3. Retrieval.enable == False 확인
        config = run.config
        has_retrieval = False
        enable_val = None
        
        # 중첩된 config dictionary에서 Retrieval 파싱
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
            # Flattened 형태로 저장되어 있을 경우 탐색
            flat_retrieval = {k: v for k, v in config.items() if "Retrieval" in k}
            if flat_retrieval:
                has_retrieval = True
                for k, v in flat_retrieval.items():
                    if k.endswith(".enable") or k == "Retrieval.enable" or "enable" in k.split("."):
                        enable_val = v
        else:
            has_retrieval = True
            if isinstance(retrieval_dict, dict):
                enable_val = retrieval_dict.get("enable")
                
        # Retrieval.enable이 명시적으로 False (혹은 None 등) 인지 확인
        is_retrieval_false = False
        if has_retrieval and enable_val in [False, "False", "false", 0, "None", None]:
            is_retrieval_false = True
        elif not has_retrieval:
            # 설정 파일에 Retrieval 자체가 없다면 기존 코드 관례에 따라 False로 간주
            is_retrieval_false = True

        if not is_retrieval_false:
            continue

        # 4. 학습이 100k 스텝까지 종료된 학습인지 확인
        # wandb 내부의 '_step'을 기준으로 합니다. (custom step 명칭이 있다면 '_step' 대신 해당 키를 사용하세요)
        step_count = run.summary.get('_step', 0) 
        
        # 스텝이 100,000 이상이며 종료(finished) 상태인 경우
        if run.state == "finished" and step_count >= 100000:
            matched_runs.append(run)
            
    if not matched_runs:
        print("조건에 맞는 run을 찾지 못했습니다.")
    else:
        print(f"총 {len(matched_runs)}개의 매칭되는 Run을 찾았습니다:")
        for idx, run in enumerate(matched_runs, 1):
            print(f"[{idx}] Run ID : {run.id}")
            print(f"    Name   : {run.name}")
            print(f"    State  : {run.state}")
            print(f"    Steps  : {run.summary.get('_step', 0)}")
            print(f"    URL    : {run.url}")
            print("-" * 50)

if __name__ == "__main__":
    find_target_runs("STORM")
