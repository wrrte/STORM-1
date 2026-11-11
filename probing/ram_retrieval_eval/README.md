# 학습 완료 seed 9999 모델의 RAM retrieval 평가

이 폴더 전체를 다른 서버의 `STORM-1/probing/ram_retrieval_eval/`로 복사한다.
기존 학습 파일, checkpoint, replay는 수정하지 않는다. 별도 프로세스에서 학습된 정책으로
게임을 진행하며, 현재까지 수집한 replay에 원래 retrieval 함수를 호출한다.
random은 같은 anchor에 대한 측정용 비교군이다. 두 모델은 eval/no-grad로 실행하며
실행 전후 state_dict SHA256이 달라지면 실패 처리한다.

## 1. 먼저 RAM API만 검사

```bash
cd /home/choemj/STORM-1
conda activate storm
python probing/ram_retrieval_eval/evaluate.py \
  --game all --ram-check-only \
  --output results/ram_verification/ram_api_check
```

`evaluate.py` 상단의 `diagnose_ram()`이 모델을 읽기 전에 검사한다.

- `getRAM()`과 `getRAM(buffer)`를 같은 에뮬레이터 상태에서 비교한다.
- 128바이트가 전부 같으면 정상 RAM으로 인정하지 않는다.
- 서로 다른 값으로 초기화한 두 버퍼에 읽어서, 일부만 쓰는 오류도 검사한다.
- 기본 API에 문제가 있고 버퍼 API가 정상이면 버퍼 API로 전환한다.
- 버퍼 overload가 없는 새 ALE에서는 정상인 기본 API를 사용한다.
- 두 API가 다르게 읽거나 모두 비정상이면 중단한다. 실제 rollout의 매 RAM 행도 검사한다.
- 게임별 `ram_check.json`에 진단, 선택한 reader, NumPy/ALE/Gymnasium 버전을 저장한다.

기본 검사에는 checkpoint가 필요 없다. 검사 후 본 실행은 새 output 폴더를 사용한다.

## 2. 기본 checkpoint와 run 설정

`runs_9999.json`은 2026-11-10에 W&B에서 확인한 세 run의 실제 config를 포함한다.
별도 W&B 로그인이나 네트워크 연결 없이 실행한다.

| 게임 | run ID | 기본 checkpoint 폴더 | value_signal | offset |
| --- | --- | --- | --- | ---: |
| Seaquest | `2esxm1h6` | `ckpt/Seaquest-myserver-9999_O` | value_diff | -3 |
| Frostbite | `0vmkg20q` | `ckpt/Frostbite-9999_value` | value | -3 |
| Pong | `36m2817w` | `ckpt/Pong-9999_retrieval` | value_diff | -3 |

기본 weights는 각 폴더의 `world_model_final.pth`, `agent_final.pth`이다.
파일이 없다고 100000-step 모델로 자동 대체하지 않는다. 다른 checkpoint를 쓰려면
`--step 100000`처럼 명시한다. W&B의 기존 점수가 그 checkpoint의 점수라고 자동으로
간주하지 않는다. 실제로 읽은 경로와 step은 metadata에 남는다.

checkpoint 최상위 경로가 다르면 `--checkpoint-root /실제/STORM-1/ckpt`를 지정한다.
폴더 이름도 다르면 `runs_9999.json`의 checkpoint 이름을 수정하거나, 한 게임에
`--checkpoint`를 지정한다. 모델 구조/가중치는 strict load하고 action 수를 검증한다.

```bash
# 한 게임의 경로를 직접 지정
python probing/ram_retrieval_eval/evaluate.py --game frostbite \
  --checkpoint /home/choemj/STORM-1/ckpt/Frostbite-9999_value \
  --device cuda:0 --output results/ram_verification/frostbite_first

# 다른 config/step 사용. 예시 경로는 실제 존재하는 경로로 바꾼다.
# --config runs/해당_run/config.yaml --step 100000
```

## 3. 충분한 값을 모으는 실행

먼저 짧은 실행으로 파일과 처리 속도를 확인할 수 있다.

```bash
python probing/ram_retrieval_eval/evaluate.py --game all \
  --device cuda:0 --steps 4096 --index-warmup 1024 \
  --output results/ram_verification/retrieval_9999_short
```

통계용으로는 여러 환경 실행 seed와 긴 rollout을 권한다. 아래 seed는 **평가 rollout
seed**이며, 학습된 파라미터의 training seed 9999는 그대로다.

```bash
python probing/ram_retrieval_eval/evaluate.py --game all \
  --checkpoint-root /home/choemj/STORM-1/ckpt \
  --device cuda:0 --precision auto \
  --steps 100000 --buffer-size 20000 \
  --seeds 2027 2028 2029 2030 2031 \
  --query-every 32 --score-batch-size 16 \
  --random-repeats 5 --bootstrap-samples 5000 \
  --output results/ram_verification/retrieval_9999_long
```

`--steps`는 action decision 수다. 한 decision은 최대 4 ALE 프레임을 진행한다.
`evaluate.py --game all`은 세 게임과 seed들을 순서대로 실행한다. `--buffer-size 20000`이면 최근 20000개
transition을 후보로 유지하면서 100000 decision 전체에 걸쳐 통계를 모은다.
64×64 RGB replay는 이 설정에서 게임/seed당 약 246MB이며 별도 RAM/결과 파일도 생긴다.
전체 데이터가 후보로 남아야 한다면 `--buffer-size 100000`으로 지정한다.
큰 객체 집합을 반복 매칭할 때 SciPy가 설치되어 있으면 빠른 backend를 사용한다.
SciPy가 없어도 포함된 NumPy Hungarian 구현으로 같은 최적 매칭을 계산한다.

세 게임을 동시에 실행하려면 `run_parallel.py`를 쓴다. 게임당 한 프로세스가 실행되고,
각 게임 안에서는 지정한 seed들을 순서대로 평가한 뒤 `summary_all_seeds.json`을 만든다.

```bash
python probing/ram_retrieval_eval/run_parallel.py \
  --devices cuda:0 \
  --steps 100000 --buffer-size 20000 \
  --seeds 2027 2028 2029 2030 2031 \
  --random-repeats 5 --bootstrap-samples 5000 \
  --output results/ram_verification/retrieval_9999_parallel
```

`--devices cuda:0`은 세 프로세스가 GPU 한 장을 공유한다. GPU 세 장에 나누려면
`--devices cuda:0 cuda:1 cuda:2`로 바꾼다. 배정 순서는 Seaquest/Frostbite/Pong이다.
GPU 한 장에서는 메모리와 연산 자원을 공유하므로 속도 향상은 실제 사용률에 따라 달라진다.
게임별 출력은 `output/logs/seaquest.log`, `frostbite.log`, `pong.log`에 저장된다.
`--dry-run`으로 실행 명령만 확인할 수 있다. Ctrl+C는 실행한 자식 프로세스도 중단한다.
기존 `--game all` 실행을 대체한다면 먼저 해당 실행을 중단하고 새 output을 지정한다.
이 실행기는 기존 seed 결과를 이어서 실행하지 않으며, 같은 결과 폴더가 있으면 시작 전에 거부한다.

`--precision auto`는 그 서버 모델 코드의 기본 AMP dtype을 사용한다. 재현 비교에는
`--precision fp32` 또는 GPU가 지원하는 `bf16`/`fp16`을 명시할 수 있다.
실제 dtype은 `metadata.json`의 `effective_model_dtype`에 저장한다.

## 4. 검색 및 비교 절차

1. 기존 환경과 같은 frameskip 4 / 마지막 두 프레임 max-pooling / 64×64 RGB 관측으로
   고정된 정책을 실행한다. RAM은 해당 관측의 마지막 원본 프레임 상태에서 읽는다.
2. 처음 `--index-warmup`개 관측에서 evaluation용 PCA/hash 인덱스를 만든다.
3. 고정 모델의 critic으로 원래 trigger 함수의 `is_warmup=True` 경로를 실행해
   evaluation EMA를 초기화한다. 기본 64 calibration batch이며 optimizer update는 없다.
4. 게임을 계속 진행하면서 현재 관측을 인덱스에 추가한다. 기본 32 decision마다
   현재 replay에서 64-frame scoring window를 추출해 원래 `add_batch_transitions()`의
   argmax/threshold/offset 규칙으로 anchor를 선택한다. episode/life-loss 경계를
   가로지르는 scoring/context window는 사용하지 않는다.
5. 각 anchor를 실제 `retrieve_contexts()`에 넣고, 반환 결과에서 anchor 자신을 제외한다.
6. 같은 시점의 같은 replay에서 유효한 context endpoint를 균등 무작위로 추출한다.
   anchor 자신을 제외하며, 각 반복의 개수는 실제 retrieval neighbor 개수와 같다.
7. 검색된 상태와 random 상태 각각의 RAM 거리와 유사도를 저장한다.

`retrieval_snapshot.py`의 두 retrieval class 본문은 제공 시점의 원본과 동일하다.
이동 편의를 위해 training_branches의 enable parser import만 evaluation의 bool=True
parser로 대체했다. source hash는 `sources.json`에 있다. 원 서버의 기존 retrieval.py는
수정하거나 덮어쓰지 않는다. 이는 과거 학습 당시의 검색 이력을 재생하는 실험이 아니라
학습 완료 모델을 고정한 새로운 online rollout에서 검색 품질을 평가하는 실험이다.
최종 weights에는 EMA/PCA/replay가 없으므로 evaluation에서 새로 구성한 사실을 기록한다.
이 평가에서 retrieval은 학습 context 검색 경로로 호출되며 정책의 행동 입력을 바꾸지 않는다.

PCA는 기본 2048 decision마다 현재 replay로 재구축한다. 최초 PCA를 고정하려면
`--rebuild-every 0`을 사용한다. 분석용 Python/NumPy/Torch RNG와 random 비교군 RNG를
정책 RNG에서 분리해, random 반복 횟수가 이후 정책 실행이나 검색 난수를 바꾸지 않게 했다.

anchor는 실제로 enqueue된 것만 평가한다. 같은 rollout step의 anchor가 반복되면
처음 한 번만 저장한다. neighbor가 없는 anchor도 queries 파일과 coverage에 포함한다.
반환 순서는 거리 순위가 아니며 CSV의 `rank`는 반환/추출 순서다.

## 5. RAM 유사도 정의

`similarity = 1 - distance`이며 0..1 범위다. 아래 그룹들의 **동일 가중 평균**을 쓴다.

| 게임 | 거리 계산 그룹 |
| --- | --- |
| Seaquest | 플레이어 / 적(상어·적 잠수함·적 미사일) / 잠수부 집합 / 탑승 인원 / 산소 |
| Frostbite | 플레이어 / 적(새·게·조개·물고기·곰) / 빙판 위치·색 / 이글루 단계 |
| Pong | 공 / 플레이어 패들 / 상대 패들 |

Frostbite 온도, Pong 공 속도·이동 방향, 누적 점수는 유사도에 넣지 않는다.
캐시된 RAM reader가 다른 scalar도 추출하더라도 metric은 이 표의 정보만 사용한다.

- 위치는 박스 중심의 정규화 L1: `0.5*(|dx|/160 + |dy|/210)`.
- 수중 잠수부와 적 등 집합은 종류/위치로 최적 매칭한다. RAM slot ID는 무시한다.
- 없는 객체 하나의 비용은 1, 두 집합이 모두 비어 있으면 0이며 max 객체 수로 나눈다.
- 빙판 x는 화면 경계를 도는 순환 거리이며 위치와 흰색/파란색 차이를 반씩 반영한다.
- 탑승 인원 차이는 6, 산소 차이는 64, 이글루 단계 차이는 16으로 정규화한다.
- 알 수 없는 RAM 해석/warning은 유사도로 강제 변환하지 않고 오류 행으로 남긴다.

이 수치는 위에서 정의한 관측 상태 기준이다. 보편적인 의미 유사도의 정답이라고
가정하지 않으며 그룹별 거리도 함께 저장해 어떤 정보 때문에 차이가 나는지 확인한다.

## 6. 저장 파일과 통계적 불확실성

한 seed면 `output/게임/`, `--seeds`를 쓰면 `output/게임/seed_숫자/`에 저장한다.

- `ram_check.json`: 동일 바이트 문제 검사 결과와 선택한 RAM API.
- `retrieval_neighbors.csv`: 실제 반환된 각 neighbor의 distance/similarity와 그룹별 거리.
- `random_neighbors.csv`: 같은 anchor의 무작위 비교군. repeat별로 같은 개수를 뽑는다.
- `pairs.jsonl`: 각 쌍의 ID, 점수, anchor RAM 128바이트, neighbor RAM 128바이트.
  ring buffer가 덮어써져도 과거 쌍을 다시 계산할 수 있다.
- `queries.csv`, `queries.jsonl`, 최종 `queries.json`: anchor별 개수와 두 방식의 평균 거리.
- `summary.json`: 최종 평균과 95% bootstrap 신뢰구간.
- `progress_summary.json`: 실행 중 주기적인 누적 평균/신뢰구간. pair/query 파일도 함께 flush한다.
- `metadata.json`: 실제 weights/config/step/seed, model hash, 변경 여부, dtype, 실행 상태.
- `examples/query_XXXXX.png`: 처음 일부 query의 원본 관측 비교. 왼쪽 retrieval / 오른쪽 random.
- `replay/*.npy`: 마지막 replay의 관측/RAM/action/reward/경계/원래 global step 정보.

평균은 먼저 query마다 neighbor 평균을 내고, query 간 동일 가중 평균을 계산한다.
완전한 retrieval/random 쌍이 없는 query는 paired mean에서 제외하되 개수와 coverage를
별도로 남긴다. 유효한 mapping이 일부만 남는 query도 paired mean에서 제외한다.

`summary.json`의 주요 필드:

- `retrieval_mean_similarity`, `random_mean_similarity`: 두 방식의 평균 유사도.
- `retrieval_mean_similarity_ci95`, `random_mean_similarity_ci95`: 각각의 95% 신뢰구간.
- `paired_distance_reduction`: random 거리 − retrieval 거리. 양수면 retrieval이 더 가깝다.
- `paired_distance_reduction_ci95`: 동일 query를 짝지어 계산한 차이의 95% 신뢰구간.
- `valid_*_pair_count`, `paired_query_count`, `anchor_episode_clusters`: 실제 표본/episode 수.
- `bootstrap_unit`, `bootstrap_clusters`: 신뢰구간에서 재표집한 단위와 그 수.

여러 seed를 합치면 **독립 rollout seed 전체를 cluster로 재표집**한다. 단일 seed에서는
anchor episode 단위로 재표집한다. 같은 episode의 수많은 neighbor 쌍을 독립 표본으로
취급해 신뢰구간을 과도하게 좁히지 않는다. 한 seed의 한 episode뿐이면 CI는 null이다.
seed 개수를 늘리면 replay를 공유하는 query들 사이의 의존성도 더 잘 다룰 수 있다.
단일 seed의 episode CI는 해당 rollout/replay에 조건부인 근사치다. 여러 학습 seed에
대한 불확실성은 이 실험으로 추정하지 않는다. 모델의 training seed는 9999로 고정된다.
중간 신뢰구간은 진행 확인용이며, 미리 정한 실행 길이/seed들의 최종 결과로 판단한다.

`--seeds` 실행이 끝나면 `output/게임/summary_all_seeds.json`도 자동 생성한다.
게임별 metric이 다르므로 서로 다른 게임의 점수를 하나의 평균으로 합치지 않는다.

별도로 실행한 seed들을 합칠 때:

```bash
python probing/ram_retrieval_eval/aggregate.py \
  --inputs results/ram_verification/retrieval_9999_long/pong/seed_* \
  --output results/ram_verification/retrieval_9999_long/pong/combined_summary.json
```

동일 seed를 중복 집계하거나 다른 모델/설정/metric을 섞으면 중단한다.
중단된 실행의 이미 flush된 부분을 분석하려면 `--allow-incomplete`를 명시한다.
이 경우 input status도 결과에 표시한다. 이전 결과 보호를 위해 evaluator는 이미
존재하는 게임/seed 폴더에 덮어쓰지 않는다. 재실행은 새 output 경로를 사용한다.

## 7. 누적 결과를 엑셀로 자동 정리

```bash
cd ~/STORM-1
python probing/ram_retrieval_eval/export_statistics.py
```

`results/ram_verification/` 아래의 평가를 재귀 검색해 **완료된 실행만** 읽고,
`results/ram_verification/retrieval_statistics.xlsx`를 생성/갱신한다. GPU는 필요 없다.
기존 raw data와 평가 파일은 읽기만 한다. 현재 storm 환경에는 필요한 패키지가 있다.
다른 환경에서 패키지가 없다면 다음 명령으로 설치한다.

```bash
python -m pip install -r probing/ram_retrieval_eval/requirements_statistics.txt
```

앞으로 완료되는 결과도 자동 반영하려면 다음 프로세스를 실행해 둔다.
60초마다 입력 상태를 검사하고 변경이 있을 때 보고서를 다시 만든다.

```bash
python probing/ram_retrieval_eval/export_statistics.py --watch 60
```

분석 폴더나 출력 파일을 직접 지정할 수도 있다.

```bash
python probing/ram_retrieval_eval/export_statistics.py \
  --inputs results/ram_verification/retrieval_9999_long \
           results/ram_verification/frostbite_pong_3seeds \
  --output results/ram_verification/three_games_statistics.xlsx
```

엑셀 시트:

- `Summary`: 게임/모델/프로토콜별 retrieval/random 유사도, seed 간 SD, 차이, 95% CI,
  양측 paired t-test와 seed 부호반전 검정 p-value, coverage와 승률.
- `Statistics`: seed 동일 가중 평균과 기존 query 동일 가중 평균을 구분한 상세 통계.
  t 구간, seed bootstrap 구간, query 가중 cluster bootstrap 구간 및 검정 정보를 포함한다.
- `Seeds`: 각 seed의 query 수, 평균, 앵커 간 표준편차.
- `Queries_001` 등: 앵커별 원자료와 점수. 제외 앵커도 사유와 함께 남긴다.
  Excel의 행 수 제한에 도달하면 다음 시트로 나눈다. `--no-queries`로 생략할 수 있다.
- `Runs`: 발견한 모든 실행과 포함 여부, 제외 사유, 입력 파일 hash.
- `Groups`: 결과를 묶는 모델/프로토콜 정보.
- `Methods`: 계산 정의, 통계 가정, 한계, 재표집 횟수, 출처.

주 추론 단위는 독립 **평가 rollout seed**다. seed 하나이면 seed 간 SD/CI/p-value는
빈칸으로 둔다. seed별 차이가 정규분포를 따른다는 가정의 t-test와, 귀무가설 아래
차이의 부호 교환 가능성을 가정하는 sign-flip 검정을 함께 기록한다. seed 3개의
exact sign-flip 검정은 양측 최소 p-value가 0.25여서 작은 t-test p와 다르게 해석해야 한다.
이는 한 학습 모델의 평가 변동이며 training seed 간 변동을 추정하지 않는다.

다른 모델·metric·retrieval 구현·precision·replay 크기·실행 길이·평가 설정은 별도
그룹으로 분리한다. 같은 그룹의 같은 seed 원자료가 동일하면 한 복사본만 포함한다.
같은 seed의 원자료가 다른 복사본들이 있으면 임의 선택하지 않고 모두 제외해
`Runs`에 표시한다. 이 경우 `--inputs`로 사용할 실행 폴더를 명시하면 된다.

기본 bootstrap은 5,000회이며, 독립 seed가 16개 이하일 때 모든 부호 조합을 열거한다.
그보다 많으면 100,000회 Monte Carlo 검정을 수행한다. 재표집 난수는 고정되어 있고
그룹별로 분리되어 있다. 여러 그룹의 p-value는 다중 비교 보정 전 값이다.

검증:

```bash
python -m unittest probing.ram_retrieval_eval.test_export_statistics -v
```

## 검증 범위

이 기기에서 세 게임의 RAM 진단을 실행했고 모두 잘못된 기본 API를 감지하여 버퍼
API로 전환했다. 로컬의 다른 학습 완료 checkpoint로 CPU 실행, 실제 retrieval/random
파일 저장, PCA rebuild, 변경되지 않은 model hash를 확인했다. 선택한 9999 weights와
원 서버의 CUDA 환경은 여기서 실행하지 않았다. 단위 테스트는 원본 학습 코드를
건드리지 않고 RAM 오류, 객체 매칭, RNG 분리, reset/ring 경계, bootstrap을 검증한다.
