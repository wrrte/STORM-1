# 목표 앵커 빈도에 가까운 TD-error only threshold 탐색

`find_td_error_threshold.py`는 기존 `measure_additive_anchors.py`를 호출해 후보별로
실제 재개 학습을 수행합니다. 기본 목표는 **5.99 anchors/update**이고 허용 오차는
0.05입니다. 현재 지원 범위는 single-env, update 주기 1, demonstration 미사용 STORM입니다.

저장된 checkpoint와 replay buffer, optimizer, EMA, RNG 상태를 매 후보마다 다시
복원합니다. threshold가 바뀌면 이후 학습 trajectory도 달라질 수 있으므로, 기존
threshold 3.5의 앵커 개수만으로 다른 threshold의 빈도를 역산하지 않습니다.

## 실행

`/home/choemj`에서 기존 100-update probe와 같은 조건으로 탐색합니다.

```bash
conda run -n storm python STORM-1/results/find_td_error_threshold.py \
  --checkpoint STORM-1/ckpt/Qbert-6010_Shared \
  --target 5.99 --steps 100 --validation-steps 1000 \
  --gpu 0 --output STORM-1/results/td_error_only_threshold_search/qbert
```

GPU 번호는 사용할 GPU에 맞게 지정합니다. `--gpu`를 생략하면 현재
`CUDA_VISIBLE_DEVICES` 설정을 상속합니다. 실행하면 GPU 재개 학습이 시작됩니다.

### GPU 0~5에서 3.5부터 0.25 간격으로 병렬 실행

아래 명령은 **실제 GPU 재개 학습**을 실행합니다. 초기 threshold는
`3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0`의 11개입니다.

```bash
cd /home/choemj
conda run --no-capture-output -n storm python STORM-1/results/find_td_error_threshold.py \
  --checkpoint STORM-1/ckpt/Qbert-6010_Shared \
  --gpus 0 1 2 3 4 5 \
  --lower 3.5 --upper 6.0 --coarse-points 11 \
  --target 5.99 --tolerance 0.05 \
  --steps 100 --validation-steps 1000 \
  --output STORM-1/results/td_error_only_threshold_search/qbert_gpus0-5
```

처음 여섯 후보는 GPU 0~5에 순서대로 배정합니다. GPU마다 한 후보씩 실행하며,
먼저 끝난 GPU가 다음 후보를 실행합니다. 여러 checkpoint를 지정하면 한 후보의
checkpoint들은 해당 GPU에서 순차 실행한 뒤 집계합니다. 각 후보에는 별도 프로세스와
`CUDA_VISIBLE_DEVICES`를 지정하므로 GPU 메모리를 서로 공유하지 않습니다.
이 GPU 번호는 호스트의 CUDA GPU ID이며, 기존 `CUDA_VISIBLE_DEVICES`는 각 자식
프로세스에서 덮어씁니다. `--gpu`와 `--gpus`는 함께 사용할 수 없습니다.

초기 11개를 모두 측정한 뒤 목표 오차를 만족하지 못하면 기본 최대 4회 더 좁은
간격으로 세분화합니다. **0.25 간격의 11개 후보만 비교**하려면
`--refinement-rounds 0`을 추가합니다. 상한을 바꿀 때는
`coarse-points = (upper - lower) / 0.25 + 1`로 맞추면 0.25 간격이 유지됩니다.

선택된 후보는 GPU 0에서 추가 1,000 update로 확인합니다.
`report.md`에 **선택된 threshold, 후보별 평균·목표와의 차이·SE·95% CI,
후속 확인 결과와 허용 오차 충족 여부**가 함께 기록됩니다.
후속 확인에서 목표를 벗어나면 그 사실을 보고하며 성공으로 표시하지 않습니다.

완료한 후보는 즉시 저장하고, 같은 명령과 출력 경로로 재실행하면 재사용합니다.
후보 실패 시 새 작업 배정을 중단하고 이미 실행 중인 후보의 종료를 기다립니다.
수정된 코드와 기존 측정 결과가 섞이지 않도록 위 예시는 새 출력 경로를 사용합니다.

### 다른 checkpoint에서 threshold 3.5 이상으로 실행

`Boxing-6040_Shared`에는 50,048 step의 전체 checkpoint, replay buffer, EMA 통계가
저장되어 있습니다. 이를 사용하는 실행 스크립트는 다음과 같습니다.

```bash
cd /home/choemj
conda run --no-capture-output -n storm bash STORM-1/results/run_td_error_threshold_boxing.sh
```

이 명령은 GPU 0~5에서 **3.5~4.0을 0.1 간격(6개)**으로 측정합니다.
`3.5, 3.6, 3.7, 3.8, 3.9, 4.0`을 GPU 0~5에 하나씩 배정하고,
각 후보를 같은 checkpoint에서 재개해 **모두 500 update씩** 측정합니다.
각 후보의 평균과 **Within-run SD across updates**를 같은 500개 앵커 수로 계산합니다.
이번 실행은 `--refinement-rounds 0`으로 추가 세분화를 하지 않습니다.
`--validation-steps 0`으로 별도의 후속 확인도 생략하므로 총 측정량은 6 × 500 updates입니다.
실행 스크립트의 하한은 **3.5로 고정**되어 있습니다.
결과는 `STORM-1/results/td_error_only_threshold_search/boxing_gpus0-5_3.5-4.0_step0.1_500updates/`의
`report.md`, `result.json`, `search.csv`에 저장합니다.

`report.md`와 `search.csv`에는 후보별 평균, Within-run SD, 목표와의 차이, SE, 95% CI가
함께 기록됩니다. Within-run SD는 0개 앵커 update를 포함한 500개 관측치의
**표본 표준편차(ddof=1)**이며, 평균의 SE와는 다른 통계입니다.
개별 `anchors.csv`와 `summary.json`도 보존됩니다.

목표에 도달하지 않아도 하한은 유지하고, 가장 가까운 실측 후보와 목표와의 차이를
보고합니다. 다른 checkpoint에서도 5.99에 가까워지는지는 실측으로 확인합니다.

스크립트 뒤에 옵션을 붙이면 하한을 제외한 기본값을 바꿀 수 있습니다. 예를 들어
Boxing에서도 상한을 8.0까지 넓히려면 새 결과 경로와 함께 실행합니다.

```bash
conda run --no-capture-output -n storm bash STORM-1/results/run_td_error_threshold_boxing.sh \
  --upper 8.0 --coarse-points 19 --refinement-rounds 4 \
  --output STORM-1/results/td_error_only_threshold_search/boxing_gpus0-5_from3.5_to8
```

먼저 **GPU 학습 없이** checkpoint·설정·최대 실행량을 확인할 수 있습니다.
아래 명령의 `--dry-run`을 제거하고 같은 `--output`으로 실제 탐색을 시작하면 됩니다.

```bash
conda run -n storm python STORM-1/results/find_td_error_threshold.py \
  --checkpoint STORM-1/ckpt/Qbert-6010_Shared \
  --output STORM-1/results/td_error_only_threshold_search/qbert --dry-run
```

여러 게임·학습 seed를 함께 사용할 때는 checkpoint를 나열합니다.

```bash
conda run -n storm python STORM-1/results/find_td_error_threshold.py \
  --checkpoint STORM-1/ckpt/Qbert-6010_Shared STORM-1/ckpt/Boxing-6040_Shared \
  --target 5.99 --steps 500 --block-size 50 --validation-steps 2000 \
  --output STORM-1/results/td_error_only_threshold_search/multiple
```

기본은 각 checkpoint에 저장된 설정을 유지합니다. **5.99를 측정했던 것과 같은
checkpoint·설정·스텝 수**를 사용해야 같은 기준의 빈도를 비교할 수 있습니다.
현재 TD-error-only campaign의 `anchor_weight_r=1` 등으로 통일하려면
`--config STORM-1/config_files/STORM.td_error_only.yaml`을 지정합니다. 이 경우
원래 probe와 다른 설정이 될 수 있습니다. 모델·입력 크기·replay-buffer 크기가
checkpoint와 달라지는 config는 거부합니다.

metadata가 없는 과거 checkpoint는 `--config`, `--env`, `--seed`로 지정합니다.
전체 checkpoint인 `training_state.pt`와 `replay_buffer.npz`가 필요하며, 모델 가중치만으로는
이 탐색을 수행하지 않습니다. 같은 게임·학습 seed의 checkpoint 중복 입력은 거부합니다.

## 탐색 방식

1. 기본 `[0, 6]` 범위를 7개 threshold로 측정합니다.
2. 측정 평균과 목표의 절대 차이가 가장 작은 후보를 고릅니다. 동률이면 작은 threshold를 선택합니다.
3. 후보 좌우에 더 좁은 간격의 threshold를 추가합니다. 기본 최대 4회 세분화합니다.
4. 허용 오차를 만족하면 세분화를 종료합니다. 허용 오차에 도달하지 않아도 가장 가까운 **실측 후보**를 저장합니다.
5. 선택한 threshold로 checkpoint를 다시 복원하고 탐색 길이만큼 학습한 뒤, 그 다음 1,000 update를 별도로 측정합니다.

`--lower`, `--upper`, `--coarse-points`, `--refinement-rounds`, `--tolerance`로 조정할 수 있습니다.
기본 최대 탐색량은 checkpoint당 **15 × 100 updates**, 후속 확인은 **100 + 1,000 updates**입니다.
`--validation-steps 0`으로 후속 확인을 생략할 수 있습니다.

학습 피드백 때문에 실측 평균이 threshold에 대해 완전히 단조적이라고 가정하지 않습니다.
이분 탐색을 사용하지 않으며, 측정하지 않은 모든 threshold 중의 전역 최적값을 보장하지 않습니다.

여러 run은 게임 내에서 동일 가중치로 평균한 뒤, 게임 간에도 동일 가중치를 줍니다.
하나라도 실패하면 그 후보를 불완전한 게임 집합으로 집계하지 않고 중단합니다.
중간 결과는 남으며 같은 명령·출력 경로로 재개할 수 있습니다. 완료된 probe는 raw CSV와
summary를 검증한 뒤 재사용하고, 중단된 probe는 새 attempt 디렉터리에서 재시도합니다.
설정·checkpoint fingerprint·코드가 바뀌면 새 출력 경로를 사용해야 합니다.

## 평균·CI·SE와 해석

- 단일 run: 연속 update의 circular block-bootstrap 95% CI와 SE.
- 동일 게임의 여러 run: run 평균을 재표집한 95% CI와 SE.
- 여러 게임: 게임별 평균을 재표집한 95% CI와 SE.
- `--block-size`, `--bootstrap-samples`, `--bootstrap-seed`로 통계 설정을 조정합니다.

탐색 결과의 CI는 threshold 선택 자체의 불확실성을 보정한 구간이 아닙니다.
후속 확인은 다른 시간 구간을 사용하지만 독립 학습 run은 아닙니다.
짧은 재개 학습에서 목표에 가까웠다는 결과가 fresh training의 `[50000,100000)`
평균도 5.99가 된다는 보장은 없습니다. 최종 학습 후 `td_error_only_anchor_stats.py`로
긴 구간의 빈도를 확인할 수 있습니다. 평균이 허용 오차 안에 들어오는지와 CI가 목표를
포함하는지는 서로 다른 조건이며, 스크립트는 평균의 절대 오차로 선택합니다.

## 결과

- `protocol.json`: checkpoint·설정·코드 fingerprint와 탐색 조건.
- `search.csv`: 실측 threshold별 평균, Within-run SD across updates, 오차, SE, 95% CI.
- `result.json`: 가장 가까운 실측 threshold, 탐색/후속 구간의 결과와 허용 오차 충족 여부.
- `report.md`: 선택된 threshold와 모든 후보의 평균·Within-run SD·SE·95% CI 및 후속 확인 결과를 정리한 표.
- `probes/`, `validation/`: 각 run의 `anchors.csv`, 실제 적용 config, summary, 실행 로그.

학습 YAML·큐·원본 checkpoint는 자동 변경하지 않습니다.

단일 threshold만 직접 측정할 수도 있습니다.

```bash
conda run -n storm python measure_additive_anchors.py \
  --checkpoint STORM-1/ckpt/Qbert-6010_Shared \
  --score td_error_only --threshold 3.0 --steps 100
```
