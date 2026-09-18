import wandb
from datetime import datetime, timedelta

def list_recent_runs_sorted():
    # WandB API 연결
    api = wandb.Api()
    entity = api.default_entity
    project_path = f"{entity}/STORM"
    
    print("8월 20일 오후 7시(KST) 이후의 학습(Run) 목록을 불러옵니다...\n")
    
    # 8월 20일 19:00 KST = 8월 20일 10:00 UTC
    # MongoDB 스타일의 createdAt 필터를 사용하여 효율적으로 검색
    filters = {"createdAt": {"$gte": "2026-08-20T10:00:00.000Z"}}
    runs = api.runs(project_path, filters=filters)
    
    # 시간 정렬을 위해 리스트에 (시간 객체, 런 이름) 형태로 임시 저장
    run_list = []
    for run in runs:
        # '2026-08-20T10:15:30Z' 형태의 UTC 시간을 datetime 객체로 파싱
        dt_utc = datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ")
        run_list.append((dt_utc, run.name))
        
    # 날짜 순(오래된 시간부터 최신 순)으로 정렬
    run_list.sort(key=lambda x: x[0])
    
    # 결과 출력
    for dt_utc, name in run_list:
        # 출력 시 보기 편하도록 다시 KST(+9시간)로 변환
        dt_kst = dt_utc + timedelta(hours=9)
        print(f"[{dt_kst.strftime('%Y-%m-%d %H:%M:%S')}] {name}")

if __name__ == "__main__":
    list_recent_runs_sorted()
