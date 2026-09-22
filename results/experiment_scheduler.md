# 결과를 보고 다음 실험 등록하기

`schedule_experiments.py`는 W&B에 접속하지 않고 분류 CSV와 현재 큐를 읽습니다.
엑셀은 사람이 확인하는 용도이며, 판정에는 반올림하지 않은 CSV 점수를 사용합니다.
Python 표준 라이브러리만 필요합니다.

```bash
cd STORM/results
python classify_wandb_runs.py
python convert_csv_to_excel.py
python schedule_experiments.py --dry-run
python schedule_experiments.py
```

마지막 명령은 실제 `STORM/job_queue_*.txt`에 명령을 추가합니다. 기존 `*_train.sh`
worker가 켜져 있으면 해당 큐에서 학습을 시작합니다. 명령은 STORM 디렉터리에서
실행하는 기존 worker를 기준으로 생성됩니다. 새 스크립트는 어느 디렉터리에서
실행해도 기본 CSV·설정·큐 경로를 동일하게 찾습니다.

## 점수와 진행 순서

- `Retrieval 미사용`과 `target: 16 (anchor 미설정)`만 점수 판정에 사용합니다.
  날짜, anchor, warmup, z-score, value/add 구분과 hash_bits=10 선호는 엑셀 기준에
  맞춥니다. baseline은 과거 점수도 포함합니다. 시드별 최신 **완료** 점수를 사용하며
  running/failed/crashed run의 중간 점수는 사용하지 않습니다.
- `update_tex.py`의 현재 `EXCLUDED_SEEDS`를 매번 읽어 평균과 비교 기준에서 제외합니다.
  제외된 시드는 체크포인트를 이미 삭제한 것으로 처리해 결과 로그와 삭제 알림을
  출력하지 않습니다. 실행 이력으로는 유지하여 새 명령에 같은 시드를 다시 쓰지 않습니다.
  이 파일이나 제외 목록을 자동 변경하지 않습니다.
- baseline 또는 target 16에서 다음 중 하나라도 만족하면 탐색 대상입니다.
  **평균 HNS가 논문보다 0.2 이상 낮거나**, **평균 HNS가 논문보다 낮으면서 같은 config의
  특정 시드 HNS가 논문보다 0.6 이상 낮은 경우**입니다. 평균이 논문 이상이면 두 번째
  조건에 해당하지 않습니다. 평균과 특정 시드 판정 모두 `EXCLUDED_SEEDS`를 제외합니다.
  평균은 각 config의 완료된 유효 시드로 계산합니다. 엑셀의 공통 시드 평균과는 다릅니다.
  두 기준은 설정의 `anomaly_hns_gap`, `anomaly_seed_hns_gap`으로 변경할 수 있습니다.
- 새 시드는 `JointTrainAgent.Retrieval.enable "['retrieval']"` 및
  `JointTrainAgent.Retrieval.save_warmup True`로 시작합니다.
  target·anchor weight·warmup steps 등 나머지 설정은 `config_files/STORM.yaml`에서
  상속하므로 생성 명령에 중복해서 붙이지 않습니다. `['retrieval']` 자체는 enable만
  활성화하며, 나머지 값을 고정하는 프리셋은 아닙니다.
- 통과 원점수는 `기존 target 16 최저 점수 + (Human - Random) × 1.0`입니다.
  HNS 1은 1%가 아닙니다. 비교 시드/점수/기준을 등록 시 저장해, 결과가 나온 뒤
  자기 자신 때문에 기준이 변하지 않게 합니다. 등록 전 이미 완료된 외부 실험을
  처음 가져올 때는 그 후보 시드를 제외한 기존 점수로 기준을 정합니다.
- 통과하면 같은 GPU 종류의 큐에
  `python -u train.py --resume_warmup ... JointTrainAgent.Retrieval.enable "['baseline']"`
  을 등록합니다. 같은 시드의 과거 baseline이 있어도 **동일 warmup**임이 확인되지
  않으면 새로 재개합니다. baseline이 crash한 경우에는 같은 warmup에서 재시도합니다.
  터미널에서는 통과한 작업의 게임 이름·시드·결과 점수·통과 문구를 녹색으로 표시합니다.
  출력을 파일이나 파이프로 보내면 색상 코드 없이 일반 텍스트로 출력합니다.
- 미달이면 warmup 삭제 필요 경로를 출력하고 다음 시드를 탐색합니다. 파일을 직접
  삭제하지 않습니다. 그 시드의 다른 실험이 큐/실행 중이면 삭제 안내도 보류합니다.
- 통과한 후보는 baseline 후속 작업을 처리합니다. 통과 시드를 이미 찾았어도 게임이
  여전히 이상 기준에 해당하면 **GPU별 18시간 작업량을 채우도록 새 시드를 추가**합니다.
  이미 시작한 병렬 후보도 결과를 판정하고 통과하면 baseline을 등록합니다.

논문 Random/Human/STORM 점수는 `convert_csv_to_excel.py`의 `REFERENCE_SCORES`를
그대로 읽습니다. 현재 예로 Gopher는 원점수 **2,154.9**, Krull은 **1,067.5** 상승이
1 HNS에 해당합니다. 통과 기준은 기존과 같이 **+1 HNS**이며, 이상 게임 판정의 0.2/0.6과
별개입니다. 새 시드는 **해당 게임**의 기존 점수·큐·상태 파일·제외 목록과 겹치지 않게
고릅니다. 다른 게임에서 사용한 시드라도 해당 게임에서 아직 사용하지 않았다면 가능합니다.

## GPU와 작업량 설정

`experiment_scheduler.json`에서 개수와 시간을 바꿀 수 있습니다.

| GPU | 개수 | warmup | retrieval 후반 | baseline 재개 | 새 target 16 총 시간 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pro6k | 1 | 1.5h | 3h | 1.5h | 4.5h |
| 3090 | 1 | 3h | 11h | 3h | 14h |
| A6000 | 4 | 2.5h | 3.5h | 2.5h | 6h |
| titan | 1 | 2h | 3h | 2h | 5h |

A6000가 추가되면 `gpus.A6000.count`를 5로, pro6k가 추가되면
`gpus.pro6k.count`를 2로 변경합니다. `count: 0`은 해당 큐에 새 작업을 배정하지 않습니다.
기존 시드는 `historical_seeds`, 새 시드 생성 규칙은 `seed_sequence`로 설정합니다.
먼저 해당 게임에서 사용하지 않은 기존 시드를 고르고, 모두 사용했다면 같은 GPU에서
다음 규칙으로 확장합니다. 기존 시드가 소진됐다는 이유로 느린 GPU로 옮기지 않습니다.

| GPU | 새 시드 확장 규칙 | 예시 |
| --- | --- | --- |
| 3090 | 2000부터 10씩 증가 | 2000, 2010, **2020**, 2030, … |
| A6000 | 6000부터 10씩 증가 | 6000, 6010, …, 6090, **6100**, … |
| pro6k | 710부터 1000씩 증가 | 기존 1·2·10 및 710, 1710, …, 3710, **4710**, … |
| titan | 9999부터 1씩 감소 | 9999, 9998, 9997, 9996, **9995**, … |

이미 상태 파일이나 큐에 GPU 배정이 기록된 시드는 그 배정을 우선합니다. 과거 버전이
생성한 10000 같은 시드도 해당 기록이 있으면 원래 GPU에서 후속 baseline을 실행합니다.
GPU별 수열은 충분히 확장하면 겹칠 수 있으므로 다른 GPU에 배정된 것으로 확인되는
시드는 건너뜁니다. 기록 없는 외부 run은 기존 시드 목록 또는 가장 가까운 수열로 GPU를
추정하며, 별도 배정이 있다면 `historical_seeds`에 명시하세요.

다른 config도 실행 시간에는 포함합니다. 예를 들어 A6000의 새 `[value, add]` 작업은
warmup 2.5h + 두 retrieval 분기 7h = 9.5h로 추정합니다. 실행 중인 분기는 경과 시간과
뒤에 남은 분기도 계산합니다. 예상 시간을 넘겼어도 CSV에 running이면 완료로 간주하지
않습니다. 중복된 기존 큐 명령은 알림을 출력하고, 대기 시간에 모두 포함합니다.

통과한 후보의 baseline을 먼저 등록합니다. 이어서 작업량이 부족한 GPU 중 예상 완료가
가장 빠른 GPU를 골라, 아직 target 16 탐색이 없는 게임부터 배정합니다. 이후에는 먼저
비는 GPU부터 추가 시드를 배정해 **각 물리 GPU의 예상 작업량을 18시간 이상**으로
채웁니다. A6000 4대도 평균이나 가장 늦게 끝나는 한 대를 기준으로 삼지 않고 각각
확인합니다. 이미 모든 GPU에 18시간 이상 작업이 있으면 신규 탐색은 추가하지 않습니다.

출력 마지막의 GPU별 예상 작업량 아래에는 **이번 호출에서 추가한 작업만** 게임별로
집계합니다. `새로운 학습`은 warmup부터 시작하는 retrieval이고, `이어지는 학습`은
저장된 warmup에서 재개하는 baseline입니다. 추가된 작업이 없는 게임은 표시하지 않습니다.

`lookahead_hours`의 기본값은 **18**입니다. 기존 호출당 16개 제한과 게임당 통과 시드
1개 제한 때문에 작업량을 덜 채우는 일이 없도록 `max_new_jobs`와
`successful_pairs_per_game`은 기본적으로 `null`(제한 없음)입니다. 원하는 경우 양의
정수로 제한할 수 있으며, 제한이나 대상 부족 등으로 18시간을 못 채우면 부족분을
보고서의 `coverage_shortfall_hours`와 안내 메시지로 표시합니다. `lookahead_hours: 0`은
게임당 탐색 하나씩 등록하는 모드입니다. 결과를 기다리며 하나만 찾으려면
`successful_pairs_per_game: 1`도 함께 설정합니다.

예를 들어 기존 작업이 전혀 없다면 pro6k는 4회 × 4.5h = 18h, 3090은 2회 × 14h = 28h,
A6000은 각 3회 × 6h = 18h, titan은 4회 × 5h = 20h를 배정합니다. 기본 GPU 개수에서
총 22회이며, 한 게임에 몰기보다 이상 게임들에 나누어 배정합니다.

이는 새 CSV를 읽는 시점의 예상 스케줄입니다. 실제 가동 상태를 실시간 조회하지 않으므로
주기적으로 결과를 갱신하고 다시 실행해야 합니다. baseline은 원래 GPU 종류에 고정하며,
그 큐를 읽는 worker에서 해당 warmup과 저장된 원래 작업 경로에 접근할 수 있어야 합니다.
같은 종류의 GPU를 여러 호스트에서 운영한다면 해당 큐의 체크포인트 저장소를 공유해야 합니다.

## 상태 유지와 복구

`experiment_scheduler_state.json`은 등록 이력, 시드/GPU, 기준 점수, 결과 run ID를
보관합니다. 반복 실행 시 이 파일을 유지하세요. `--dry-run`은 큐와 상태 파일을 쓰지
않으며, 동시 실행 조정을 위한 잠금 파일은 생성할 수 있습니다.

작업이 큐에서 사라졌지만 CSV에 아직 나타나지 않으면 `awaiting_result`로 예약을
유지합니다. W&B 지연, killed run의 CSV 누락, 기록 실패가 있을 때 비싼 실험이 중복
실행되는 것을 막습니다. 실제 작업이 종료·실패한 것을 확인한 경우 다음처럼 해제합니다.

```bash
python schedule_experiments.py --fail-job Gopher:4710:retrieval
```

실행 중이거나 큐에 남은 작업은 이 옵션으로 해제할 수 없습니다. 기존 worker와 동일한
`job_queue_*.lock`을 잡은 동안 명령을 추가합니다. 상태를 먼저 저장하므로, 파일 쓰기
도중 장애가 나면 누락된 명령을 자동 중복 발송하지 않고 `awaiting_result`로 남깁니다.
다른 CSV나 복제 작업 폴더를 사용한다면 큐와 상태 파일도 함께 분리하세요.

```bash
python schedule_experiments.py --games Gopher Krull --dry-run
python schedule_experiments.py --dry-run --report /tmp/storm-plan.json
python schedule_experiments.py --help
```

`--games`는 신규 탐색 대상만 제한합니다. 이미 상태 파일에 기록된 작업의 결과와 후속
baseline은 계속 처리합니다. 이전 CSV에 warmup 경로가 없으면 통과 판정은 기록하지만
경로를 추측해 baseline을 등록하지 않습니다. 갱신된 `classify_wandb_runs.py`를 실행하면
실행 인자에서 경로를 읽어 `Warmup Directory`, `Base Run Name`, `Training Phase`를
내보냅니다. 경로가 계속 없다면 해당 run의 W&B metadata를 확인해야 합니다.

테스트는 학습을 실행하지 않고 임시 큐·CSV로 순차적인 상태 변화와 중복 방지를 검증합니다.

```bash
python -m unittest discover -s STORM/results -p test_experiment_scheduler.py
```
