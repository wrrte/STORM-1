# STORM anchor 선택 빈도 비교

`anchor_frequency.py`는 W&B의 `choemj-kaist/STORM` 기존 run을 읽어 Full FLASH
(`score_combination=multiply`, threshold 3.5)와 additive (`add`, threshold 4.0)를
비교한다. 학습이나 W&B 데이터 수정은 수행하지 않는다.

## 학습에서 저장하는 관련 그래프

| W&B metric | 의미와 기록 조건 | threshold 비교에 쓰는 방법 |
|---|---|---|
| `Retrieval/triggered_anchors_step` | world-model update마다 새로 anchor queue에 들어간 수. warmup은 0. | **주 분석 지표** |
| `Retrieval/retrieved_contexts` | actor imagination에 추가된 anchor와 이웃 context의 총수. retrieval 결과가 있을 때만 기록. | anchor 수로 사용할 수 없음 |
| `Retrieval/active_anchors_queue` | retrieval을 수행한 뒤 queue에 남은 anchor 수. retrieval 결과가 있을 때만 기록. | queue 잔량이며 신규 선택 수가 아님 |
| `Retrieval/candidates_before_max` | 최종 context 수 제한 적용 전 후보 context 수. 결과가 있을 때만 기록. | 이웃 수·hash 상태에도 영향받음 |
| `Retrieval/lazy_rebuild_hit_rate` | lazy hash 재검증에서 유지된 일치 비율. 결과가 있을 때만 기록. | hash 상태 확인용 |
| `Retrieval/avg_bucket_size`, `num_active_buckets`, `bucket_size_hist` | hash bucket 평균 크기, 수, histogram. retrieval 결과가 있을 때 기록. | memory 상태 확인용 |
| `Retrieval/{signal}_mean`, `{signal}_var` | TD error 및 사용 중인 value signal의 EMA 평균·분산. 활성화된 retrieval의 world-model update마다 기록. | 결합 score의 분포가 아니므로 threshold별 확률을 복원할 수 없음 |
| `Retrieval/global_rebuild_triggered` | hash 전체 rebuild 여부. | rebuild 시점 확인용 |
| `Retrieval/warmup_ended_at_step`, `shared_warmup_ended_at_step`, `dynamic_warmup_*` | 해당 경로에서 warmup 관련 사건이 발생할 때 기록. | 학습 시간축 확인에 도움 |

로깅 위치는 `train.py`의 `train_world_model_step`, `world_model_imagine_data`,
world-model 학습 block이다. `utils.py:Logger`는 metric을 모았다가 iteration마다
`wandb.log()` 한 번으로 내보낸다. W&B `_step`은 이 호출 번호이고 환경 step과 다르다.
TensorBoard의 metric별 step 역시 환경 step이 아니다.

선택 로직은 STORM 저장소의 상위 파일 `../retrieval.py:add_batch_transitions`에 있다.
각 sampled sequence의 최대 score가 threshold 이상이고 anchor의 hash key가 존재해야
queue에 추가된다. 동일한 replay 상태가 다시 선택될 수 있으므로 이 metric은
**중복을 허용한 anchor 삽입 횟수**다. 순수 threshold 통과 개수, 고유 anchor 수,
실제로 actor가 사용한 anchor 수는 현재 로그로 직접 알 수 없다.

## 실행

저장소 루트에서 기존 `storm` conda 환경을 사용한다. W&B, NumPy, Matplotlib가 필요하다.
저장소의 `venv`에는 Matplotlib가 없을 수 있다.

```bash
conda run -n storm python results/anchor_frequency.py
```

기본은 `scan_history()`를 사용하고, 실패하거나 기록 수 검증을 통과하지 못하면
작은 구간별 GraphQL 조회로 전환한다. 서버가 `history(samples=1000000)` 요청에도
10,000개만 반환하는 것을 확인했으므로, 큰 samples 값만으로 전체 기록이라고 판단하지 않는다.
fallback에서는 5,000 step씩 조회하고 경계 중복을 제거한다.
전체 학습에서 예상되는 update 수와 연속된 `_step`을 모두 확인하며,
샘플링·누락이 있으면 해당 run을 실패로 처리한다. 누락을 0으로 채우지 않는다.

```bash
# 로컬 raw 캐시로 집계와 그래프를 재생성: W&B 연결 불필요
conda run -n storm python results/anchor_frequency.py --offline

# 서버의 최신 run 목록과 history를 다시 수집
conda run -n storm python results/anchor_frequency.py --refresh

# 특정 게임·시드만 별도 출력 디렉터리에서 분석
conda run -n storm python results/anchor_frequency.py \
  --games Pong --seeds 9999 --output results/anchor_frequency_pong

# 캐시된 구간별 history와 scan_history 전체 기록을 행 단위로 대조
conda run -n storm python results/anchor_frequency.py \
  --verify-scan 36m2817w olddbu1z

# 데이터 검증 테스트: 네트워크 불필요
conda run -n storm python -m unittest discover -s results -p test_anchor_frequency.py -v
```

W&B 인증은 `WANDB_API_KEY`, 저장소의 `.wandb_api_key`, SDK의 기존 인증 순으로 사용한다.
키는 출력하거나 결과 파일에 복사하지 않는다. `wandb.init()`을 호출하지 않는다.

## 비교 기준과 계산식

- 완료된 run 중 target 16, fixed warmup 50,000, `value_diff`, hash retrieval,
  `batch_size_reduction=retrieved`, 명시적인 고정 anchor weight 0.0625만 사용한다.
  상대 가중치 `anchor_weight_r`를 사용하는 실험은 제외한다.
- 같은 게임·실제 CLI 학습 시드·모델 설정·학습 설정·retrieval 설정끼리 매칭한다.
  `anchor_offset=-2`와 `-3`은 합치지 않는다. 시드는 run 이름과 실행 인자에서 확인한다.
  W&B의 `BasicSettings.Seed=0`은 실제 실행 시드와 다를 수 있다.
- 과거 run에서 `score_combination`, `value_signal`, `neighbor_source`, `anchor_weight_r`가
  기록되지 않았으면 각각 `multiply`, `value_diff`, `retrieval`, `None`을 적용한다.
  anchor weight 자체가 기록되지 않은 run은 제외한다. 시드·설정이 같아도 다른 commit과
  학습 trajectory 차이는 남을 수 있다. commit과 checkpoint 경로를 CSV에 보존한다.
- 후보가 여러 개면 compatible additive run이 최신인 쌍을 먼저 선택하고, 그 안에서
  Full FLASH가 최신인 쌍을 선택한다. reward나 anchor 빈도로 run을 고르지 않는다.
- 현재 코드는 single-env, dynamics/actor update 주기 1, demonstration 미사용인
  STORM 학습을 대상으로 한다. 다른 주기는 임의 환산하지 않고 제외한다.
- 기본 요청 구간은 환경 step `[50000, 100000)`이다. 각 pair의 두 run이 모두
  warmup을 끝내고 기록을 가진 공통 구간을 쓴다. shared warmup이 episode 경계까지
  연장될 수 있어 실제 비교 시작점은 50,000보다 클 수 있다.
- 재개 run은 metadata의 `--resume_from .../shared_warmup_N`에서 N을 읽고,
  새 학습은 `BufferWarmUp`에서 시작한다. 전체 예상 update 수 검증 후 순서대로 환경
  step을 붙인다. 비교 CSV는 실제 사용한 반열린 구간을 보존한다.

```text
anchors_per_update = sum(triggered_anchors_step) / number_of_updates
selected_sequence_fraction = sum(triggered_anchors_step) / (updates * BatchSize)
updates_with_any_anchor_fraction = count(triggered_anchors_step > 0) / updates
anchors_per_1000_env_steps = sum(triggered_anchors_step) / environment_steps * 1000
```

게임 내 paired seed 평균을 먼저 계산하고 게임을 동일 가중치로 평균한다.
ratio의 95% bootstrap CI는 게임별 두 설정의 평균을 함께 재표집해 계산한다.
학습 update를 독립 표본으로 취급하지 않는다. 기본 ±10%는 기술적인 비교 기준일 뿐,
리뷰에서 동등성을 주장하려면 허용 차이를 별도로 정당화해야 한다.

## 결과 파일

기본 출력은 `results/anchor_frequency/`다.

- `report.md`, `summary.json`: 전체 결과와 해석상의 한계.
- `per_run.csv`, `pairs.csv`, `per_game.csv`: run, 게임·시드 pair, 게임별 수치.
- `frequency_comparison.png`, `.pdf`: 게임별 선택 빈도와 paired-seed 산점도.
- `frequency_curves.png`, `timeseries.csv`: 5,000 환경 step 구간별 빈도.
- `discovery.json`, `raw/<run_id>/run.json`, `anchors.csv.gz`: 조회 당시 설정·metadata·전체 anchor history.
- `selection_audit.json`, `failures.json`: 매칭되지 않은 run, 중복 후보, 조회·검증 실패.
- `scan_verification.json`: `--verify-scan`으로 대조한 전체 history의 일치 여부.

조회·검증 실패가 있으면 완료된 pair만 보고서에 포함하고 종료 코드 2를 반환한다.
미매칭 설정은 별도 audit에 남긴다. 성공한 데이터도 보고서와 함께 범위를 확인해야 한다.

## threshold 선택의 해석

`x = ReLU(z_TD)`, `y = softplus(z_value)`라 두면 두 score는 `xy`와 `x+y`다.
두 score 사이에 전역적인 대소 관계가 없으므로, additive threshold가 반드시 더 커야
하거나 정확히 4.0이면 같은 빈도가 된다는 결론은 수식만으로 나오지 않는다.
예를 들어 `(x,y)=(0,4)`에서는 add만 통과하고 `(1.9,1.9)`에서는 multiply만 통과한다.
기존 로그에서 비슷한 빈도가 확인되면 경험적 근거로 쓸 수 있지만, score의 공동 분포를
저장하지 않았으므로 기존 로그만으로 다른 additive threshold의 빈도를 재계산할 수는 없다.

W&B 전체 history의 API 동작은 [공식 SDK 소스](https://github.com/wandb/wandb/blob/main/wandb/apis/public/runs.py)에 설명되어 있다.
