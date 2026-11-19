# TD-error only: 스텝당 앵커 수, CI, SE

학습 중 저장된 TensorBoard 로그를 읽습니다. GPU 학습이나 W&B 연결 없이 실행합니다.
필요 패키지는 기존 `storm` 환경의 NumPy, PyYAML, TensorBoard입니다.

STORM-1 디렉터리에서:

```bash
python results/td_error_only_anchor_stats.py
```

기본값은 전용 campaign의 완료된 run이며, 각 게임의 마지막 완료 attempt를 선택합니다.
평가 점수나 앵커 수로 run을 선택하지 않습니다. run의 실제 저장 설정을 확인하고,
서로 다른 campaign은 별도로 집계합니다.

```bash
# 특정 게임과 threshold
python results/td_error_only_anchor_stats.py --games Alien Amidar --threshold 3.5

# 구간과 bootstrap 설정 변경
python results/td_error_only_anchor_stats.py \
  --start 50000 --stop 100000 --block-size 1000 \
  --bootstrap-samples 10000 --confidence 0.95 --bootstrap-seed 0

# 실행 중인 run의 현재 관측 구간도 포함
python results/td_error_only_anchor_stats.py --include-running

# 이벤트 파일을 다시 읽어 캐시 갱신
python results/td_error_only_anchor_stats.py --refresh
```

`--manifest PATH`, `--root PATH`, `--output PATH`로 입력과 출력 경로를 지정할 수 있습니다.
캐시는 이벤트 파일의 경로·크기·수정 시각과 설정 파일의 SHA256이 같을 때 재사용합니다.

## 무엇을 한 스텝으로 세는가

분자는 `Retrieval/triggered_anchors_step`의 합입니다. 이 값은 각 world-model update에서
threshold와 hash-key 조건을 만족해 **앵커 큐에 새로 삽입된 횟수**입니다. 같은 replay
상태의 반복 삽입도 셉니다. 앵커가 없는 update의 0도 분모에 포함합니다.

현재 campaign은 `NumEnvs=1`, `TrainDynamicsEverySteps=1`이므로 world-model update
1회가 환경 스텝 1회에 대응합니다. 다른 주기·병렬 환경·resume run은 이 스크립트에서
거부합니다. TensorBoard의 metric step은 환경 step이 아니라 metric별 0-based counter입니다.
현재 fresh-run 경로에서는 `env_step = BufferWarmUp + metric_step`으로 복원합니다.
demonstration을 사용하는 Freeway도 이 fresh-run 경로에서는 같은 매핑을 사용합니다.

기본 분석 구간은 `[Retrieval.warmup_steps, FinalEvalStep)`입니다. 현재 설정에서는
`[50000, 100000)`의 50,000개 update입니다. 따라서 warmup의 0으로 평균을 희석하지
않으며, 100k checkpoint 이후의 학습도 제외합니다. `--start`를 지정하면 warmup의
관측된 0까지 포함할 수 있습니다. `BufferWarmUp` 이전에는 world-model update가 없습니다.

모든 이벤트를 순회하며 대상 scalar만 추출합니다. step 중복·reset·누락, 비정수/음수
앵커 수, 완료 run의 예상 update 수 불일치는 오류로 처리하며 누락을 0으로 채우지 않습니다.
`--include-running`은 실제 관측된 구간까지만 사용합니다. 이때 run마다 구간 길이가
다를 수 있으며 `complete_window`와 실제 시작·종료 step을 결과에 기록합니다.

## 평균과 불확실성의 단위

평균은 `sum(anchor_count) / number_of_updates`입니다.

- **run별**: 기본 길이 1,000 update의 circular moving-block bootstrap을 사용합니다.
  연속 update를 묶어 재표집하며, 필요하면 마지막 block을 잘라 원래 관측 수를 유지합니다.
  관측 구간이 짧으면 block 길이는 관측 수의 절반까지 자동 축소합니다.
- **게임별**: run별 평균을 동일 가중치로 평균합니다. CI와 SE는 독립 run 평균을
  재표집합니다. run이 하나이면 이 독립-run CI와 SE는 빈 값이며, 해당 run의
  시간축 기준 CI와 SE는 `per_run.csv`에 있습니다.
- **전체**: 게임별 평균을 동일 가중치로 평균합니다. CI와 SE는 **게임**을 재표집해
  계산하므로 학습 스텝 수가 많은 게임에 더 큰 가중치를 주지 않습니다.

모든 bootstrap에서 `SE = std(bootstrap_means, ddof=1)`이고, 기본 95% CI는
bootstrap 평균의 2.5%, 97.5% 분위수입니다. CI는 임의로 `평균 ± SE`로 계산하지 않습니다.
반복 수와 난수 seed를 저장해 재현할 수 있습니다.

run별 CI는 관측된 학습 trajectory의 시간적 재표집에 대한 근사 구간입니다. 학습의
비정상성과 block 길이에 영향을 받습니다. 여러 독립 학습 seed의 변동성을 뜻하지
않습니다. 전체 CI도 게임 간 변동에 대한 구간이며, 고정된 benchmark의 독립 학습
seed 불확실성과 구분해야 합니다. block 길이 민감도는 예를 들어 500/1000/2000으로
바꾸어 확인할 수 있습니다. 독립 단위가 하나이면 해당 CI와 SE는 계산하지 않습니다.

## 출력

기본 경로는 `results/td_error_only_anchor_stats/`입니다.

- `per_run.csv`: run별 평균, SE, CI 하한/상한, SD, 관측 수, 총 앵커 삽입 수, 실제 구간.
- `per_game.csv`: 게임별 평균과 독립 run 기준 CI/SE.
- `summary.json`: campaign별 전체 평균과 CI/SE, 설정, source fingerprint, 제외/실패 내역.
- `report.md`: 전체 결과와 집계 단위 설명.
- `cache/*.npz`: 추출한 metric step과 앵커 수. pickle을 사용하지 않습니다.

로그 검증 실패나 분석할 run이 없으면 종료 코드 2를 반환합니다. 아직 시작하지 않은
작업과 기본 설정에서 제외된 실행 중 작업은 `summary.json`의 `skipped`에 기록합니다.

검증:

```bash
python -m unittest test_td_error_only_anchor_stats -v
```
