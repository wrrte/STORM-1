import os
import wandb

def main():
    # 1. 자동 로그인 처리
    api_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.wandb_api_key')
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as f:
            api_key = f.read().strip()
            wandb.login(key=api_key)
            print("✅ .wandb_api_key를 사용하여 성공적으로 로그인했습니다.\n")
    else:
        print("⚠️ .wandb_api_key 파일을 찾을 수 없습니다. 기존 환경의 로그인 정보를 사용합니다.\n")

    api = wandb.Api()
    project_path = "choemj-kaist/STORM"
    
    run_name_1 = "Qbert_p0086jqt_3710_O"
    run_name_2 = "Qbert_8yv3yf7e_3710_O"
    
    print(f"'{project_path}' 프로젝트에서 두 런({run_name_1}, {run_name_2})의 정보를 가져옵니다...")
    
    runs = api.runs(project_path)
    
    run1 = None
    run2 = None
    
    # 런 이름으로 검색 (이름이 완전히 동일한 경우 최신 런을 가져오도록 구현됨)
    for r in runs:
        if r.name == run_name_1 and not run1:
            run1 = r
        elif r.name == run_name_2 and not run2:
            run2 = r
            
        if run1 and run2:
            break
            
    if not run1:
        print(f"❌ {run_name_1} 런을 찾을 수 없습니다.")
        return
    if not run2:
        print(f"❌ {run_name_2} 런을 찾을 수 없습니다.")
        return
        
    print(f"\n========================================")
    print(f"[기본 정보]")
    print(f"Run 1: {run1.name}")
    print(f"  - ID: {run1.id}")
    print(f"  - State: {run1.state}")
    print(f"  - Created At: {run1.created_at}")
    print(f"  - Commit: {run1.commit}")
    
    print(f"\nRun 2: {run2.name}")
    print(f"  - ID: {run2.id}")
    print(f"  - State: {run2.state}")
    print(f"  - Created At: {run2.created_at}")
    print(f"  - Commit: {run2.commit}")
    print(f"========================================\n")
    
    print("[Config (하이퍼파라미터) 차이점]")
    config1 = run1.config
    config2 = run2.config
    
    # 두 런의 config 키들의 합집합을 구하여 모든 설정 비교
    all_keys = set(config1.keys()).union(set(config2.keys()))
    diff_count = 0
    
    # 중첩된 dictionary 형식을 비교하기 좋게 펼치는 함수(flatten) 없이 단순 비교
    # 보통 wandb config는 펼쳐져서 저장되거나 dict 형태 그대로입니다.
    for key in sorted(all_keys):
        val1 = config1.get(key, '<설정 없음>')
        val2 = config2.get(key, '<설정 없음>')
        
        if val1 != val2:
            print(f" 🔹 {key}")
            print(f"    - {run1.name} : {val1}")
            print(f"    - {run2.name} : {val2}")
            diff_count += 1
            
    if diff_count == 0:
        print("  ✅ Config 설정이 완벽히 동일합니다.")
    else:
        print(f"\n  총 {diff_count}개의 설정 값 차이가 발견되었습니다.")
        
    print(f"\n========================================")
    print("[주요 결과 지표(Summary) 비교]")
    keys_to_compare = [
        'eval/episode_avg_return', 
        'eval/episode_avg_length', 
        '_step', 
        '_runtime'
    ]
    
    for key in keys_to_compare:
        val1 = run1.summary.get(key, 'N/A')
        val2 = run2.summary.get(key, 'N/A')
        
        print(f" 🔹 {key}")
        print(f"    - {run1.name} : {val1}")
        print(f"    - {run2.name} : {val2}")

if __name__ == "__main__":
    main()
