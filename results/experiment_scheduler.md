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
  이 파일이나 제외 목록을 자동 변경하지 않습니다.
- baseline 또는 target 16의 시드 평균이 논문보다 **1 HNS 이상 낮으면** 탐색 대상입니다.
  평균은 각 config의 완료된 유효 시드로 계산합니다. 엑셀의 공통 시드 평균과는 다릅니다.
- 새 시드는 `JointTrainAgent.Retrieval.enable "['retrieval']"` 및
  `JointTrainAgent.Retrieval.save_warmup True`로 시작합니다.
- 통과 원점수는 `기존 target 16 최저 점수 + (Human - Random) × 1.0`입니다.
  HNS 1은 1%가 아닙니다. 비교 시드/점수/기준을 등록 시 저장해, 결과가 나온 뒤
  자기 자신 때문에 기준이 변하지 않게 합니다. 등록 전 이미 완료된 외부 실험을
  처음 가져올 때는 그 후보 시드를 제외한 기존 점수로 기준을 정합니다.
- 통과하면 같은 GPU 종류의 큐에
  `python -u train.py --resume_warmup ... JointTrainAgent.Retrieval.enable "['baseline']"`
  을 등록합니다. 같은 시드의 과거 baseline이 있어도 **동일 warmup**임이 확인되지
  않으면 새로 재개합니다. baseline이 crash한 경우에는 같은 warmup에서 재시도합니다.
- 미달이면 warmup 삭제 필요 경로를 출력하고 다음 시드를 탐색합니다. 파일을 직접
  삭제하지 않습니다. 그 시드의 다른 실험이 큐/실행 중이면 삭제 안내도 보류합니다.
- 기본 탐색 목표는 게임당 통과 시드 **1개**입니다. 통과하면 추가 시드 탐색을 멈추고
  baseline 후속 작업을 처리합니다. 이미 시작한 병렬 후보도 결과를 판정하고 통과하면
  baseline을 등록합니다. baseline 결과가 낮다는 이유만으로 탐색을 자동 재개하지는
  않습니다. 더 많은 통과 시드를 찾으려면 아래 `successful_pairs_per_game`을 늘리세요.

논문 Random/Human/STORM 점수는 `convert_csv_to_excel.py`의 `REFERENCE_SCORES`를
그대로 읽습니다. 현재 예로 Gopher는 원점수 **2,154.9**, Krull은 **1,067.5** 상승이
1 HNS에 해당합니다. 새 시드는 기존 점수·큐·상태 파일·제외 목록과 겹치지 않게 고릅니다.

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
기존 시드의 GPU 종류는 `historical_seeds`, 새 시드는 GPU마다 겹치지 않는
`new_seed_range`로 식별합니다. 기존 titan 큐에 있는 9996도 등록해 두었습니다.

다른 config도 실행 시간에는 포함합니다. 예를 들어 A6000의 새 `[value, add]` 작업은
warmup 2.5h + 두 retrieval 분기 7h = 9.5h로 추정합니다. 실행 중인 분기는 경과 시간과
뒤에 남은 분기도 계산합니다. 예상 시간을 넘겼어도 CSV에 running이면 완료로 간주하지
않습니다. 중복된 기존 큐 명령은 알림을 출력하고, 대기 시간에 모두 포함합니다.

먼저 게임마다 실행·대기 중인 target 16이 없을 때 하나를 배정하며, 예상 완료가 가장
빠른 GPU를 고릅니다. 다음으로 `lookahead_hours` 안에 작업이 비는 것으로 예상되는
GPU에 추가 시드를 배정합니다. 이 추가 배정에서는 먼저 비는 GPU부터 채웁니다.
기본값은 6시간이고, 0이면 추가 병렬 탐색 없이 게임당 하나씩 결과를 기다립니다.
호출당 추가 명령은 `max_new_jobs`(기본 16)개까지입니다. 통과한 후보의 baseline을 먼저
등록합니다. `successful_pairs_per_game`(기본 1)은 탐색을 멈출 통과 시드 수입니다.

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
python schedule_experiments.py --fail-job Gopher:10000:retrieval
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
