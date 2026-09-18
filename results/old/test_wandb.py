import wandb

def test_wandb_history():
    api = wandb.Api()
    entity = api.default_entity
    project_path = f"{entity}/STORM"
    
    run_name = "Frostbite_76uw5jse_3710_O"
    print(f"[{run_name}] 정보를 검색합니다...")
    
    runs = api.runs(project_path, filters={"display_name": run_name})
    if len(runs) == 0:
        print("해당 이름의 Run을 찾을 수 없습니다.")
        return
        
    run = runs[0]
    
    print("\n--- [Method 3] run.history(samples=1000000) 사용 (샘플링 방지) ---")
    try:
        # samples를 아주 크게 주어 백엔드의 샘플링을 비활성화하고 모든 기록을 가져옵니다.
        df = run.history(keys=["Retrieval/retrieved_contexts"], samples=1000000)
        print(f"전체 기록 개수: {len(df)}")
        if not df.empty and "_step" in df.columns:
            # 가져온 전체 데이터 중 가장 작은 _step 찾기
            first_step = int(df["_step"].min())
            print(f"\n=> 샘플링 없이 전체에서 찾은 첫 _step: {first_step}")
            print(f"=> 우리가 원하는 값(첫 _step + 1024): {first_step + 1024}")
        else:
            print("데이터가 비어있거나 '_step' 열이 없습니다.")
    except Exception as e:
        print("history() 에러:", e)

if __name__ == "__main__":
    test_wandb_history()
