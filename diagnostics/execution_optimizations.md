# STORM 실행 최적화 — 2026-09-30

현재 `config_files/STORM.yaml`은 기존 두 옵션과 아래 세 옵션을 모두 활성화한다.
기존 Hero 실행 명령을 새 프로세스로 시작하면 적용된다. 실행 중인 프로세스를 재시작하지 않았다.
설정에 새 항목이 없는 이전 YAML에는 `False / legacy / False`가 적용된다.
공통 warmup에서 저장하는 설정 및 후속 실험 분기에도 옵션이 전달된다.

```yaml
Performance:
  VectorizedReplaySampling: True
  DisableDistributionValidation: True
  BatchScalarLogging: True
  RetrievalStatisticsMode: "vectorized"
  ProjectedKVCache: True
```

## 구현과 동일성

- `BatchScalarLogging`: 월드 모델 8개, actor-critic 6개의 손실·통계 값을 dtype/device별로 묶어 읽는다.
  계산식, 각 값의 정밀도, 로그 순서·빈도를 유지한다. 한 번의 전송당 스칼라 여러 개를 읽으며,
  mixed dtype를 낮은 정밀도로 강제 변환하지 않는다. 테스트한 학습 상태는 비트 단위로 일치했다.
- `RetrievalStatisticsMode: legacy`: 기존 reduction 및 개별 `.item()` 경로.
- `RetrievalStatisticsMode: batched`: 기존 마스킹과 개별 `mean`/`var(unbiased=False)`를 유지하고,
  결과 전송만 묶는다. NumPy float64 EMA를 기존 순서로 갱신한다. 비트 보존용 경로다.
- `RetrievalStatisticsMode: vectorized`: 환경별 유효 마스크를 사용한 고정 크기 GPU reduction으로
  평균·모집단 분산을 구하고 한 번에 CPU로 옮긴다. GPU boolean 조건문과 동적 boolean gather를
  통계 함수에서 제거한다. EMA 상태와 갱신식은 CPU float64로 유지해 기존 체크포인트와 호환된다.
  reduction 순서가 달라지므로 수학적 동일성을 목표로 하며 비트 동일성을 보장하지 않는다.
  빈 환경은 갱신하지 않고, 단일 표본 분산은 0이며, 제외된 NaN은 통계에 반영하지 않는다.
- `ProjectedKVCache`: eval/no_grad 경로에서만 새 토큰을 K/V로 투영하여 미리 할당한 캐시에 기록한다.
  매 rollout 시작에 캐시를 초기화한다. attention은 dropout=0인 SDPA를 사용하며,
  현재 query가 모든 과거 key를 볼 수 있도록 `is_causal=False`를 지정한다.
  필요한 위치 임베딩만 직접 읽는다. projection/reduction 커널과 반올림이 달라질 수 있으므로
  수학적 동일성 범주다. training/gradient 경로는 기존 구현을 사용한다.

난수 샘플링의 호출 수·순서·shape는 유지한다. Context의 사용하지 않는 latent sample도 삭제하지 않는다.
수치 차이로 categorical action이나 retrieval threshold 결과가 달라질 수 있으므로,
최종 trajectory·점수의 동일성을 의미하지 않는다. 캐시와 성능 옵션은 모델 state_dict에 추가되지 않는다.
기존 모델·옵티마이저·리트리벌 체크포인트 스키마를 유지한다.

GradScaler의 NaN/Inf 처리, EMA 갱신 주기, 환경 수, 배치 크기, 학습 주기와 rollout 길이는 유지한다.

비트 보존 경로만 선택하려면 기존 학습 명령에 다음을 추가한다.

```bash
Performance.BatchScalarLogging True Performance.RetrievalStatisticsMode batched Performance.ProjectedKVCache False
```

새 옵션만 비활성화하고 기존 두 최적화를 기준선으로 사용하려면 다음을 추가한다.

```bash
Performance.BatchScalarLogging False Performance.RetrievalStatisticsMode legacy Performance.ProjectedKVCache False
```

## 모델 업데이트와 캐시의 경계

`train.py`의 학습 순서는 월드 모델 업데이트 → imagination → actor-critic 업데이트다.
각 `imagine_data()` 호출은 시작할 때 `reset_kv_cache_list()`로 이전 캐시와 position을 비운다.
그 후 replay 관측을 **업데이트된 encoder**로 다시 인코딩하고, 같은 모델 가중치로
context 및 미래 궤적의 K/V를 만든다. 한 imagination 호출 안에서는 optimizer step이 없다.
다음 월드 모델 업데이트 이후의 imagination도 다시 reset과 재인코딩을 거친다.
온라인 행동 선택의 context 재인코딩 및 학습용 transformer full forward도 기존 경로다.

기존 구현도 한 rollout 안에서 과거 feature를 저장했다. 차이는 그 feature에 대한
K/V projection을 매번 반복했는지, 고정된 같은 가중치에서 한 번 계산해 저장하는지다.
현재 학습 경로에서 이전 모델 버전의 K/V나 latent를 다음 모델 버전에 넘겨 쓰는 근사는 없다.
이 동일성 주장은 현재의 reset/업데이트 순서를 전제로 한다.

추가 회귀 `test_optimizer_updates_cannot_reuse_previous_rollout_kv`는 이전 K/V 전체를
NaN으로 채운 뒤 실제 월드 모델 optimizer 업데이트를 2회 수행하여 각각 다음을 검사한다.

- K projection 가중치가 실제로 변경됨.
- 다음 imagination 시작에서 reset이 호출되고 이전 캐시와 다른 저장 공간을 사용함.
- 새 context/rollout 출력과 사용한 K/V 구간이 모두 finite임.
- 한 imagination 전후 월드 모델과 agent의 파라미터 및 BatchNorm 등 buffer가 비트 단위로 동일함.

추가 회귀를 포함한 실행 최적화 집중 테스트 8개가 통과했다.

## 검증

최종 STORM 테스트 16개와 공유 리트리벌 테스트 86개가 모두 통과했다.

- 기존 모델의 연속 2회 업데이트에서 배치 로깅 전후 출력, 행동, 손실, 기울기,
  파라미터, optimizer, GradScaler, 반환값 EMA, Python/NumPy/Torch 난수 상태를 비트 단위로 비교.
- 통계 legacy/batched의 비트 비교와 vectorized의 수치 비교: CPU/CUDA, FP32/FP64,
  여러 환경, 빈 환경, 단일 표본, 제외된 demonstration/NaN, 반복 EMA 갱신, 이전 형식 checkpoint 복원.
- Retrieval이 이전 EMA로 점수를 계산한 뒤 현재 배치를 반영하는 순서와 threshold에서 떨어진 anchor 선택 비교.
- KV 캐시: 고정된 입력의 24-step 비교를 FP64, FP32, BF16, FP16에서 수행.
  FP64/FP32 허용 오차는 각각 atol/rtol 1e-10, 3e-6이며, BF16은 .05/.03, FP16은 .008/.008이다.
  배치 크기 변경·reset·새 가중치 로딩, training fallback, RNG 유지와 실제 모델 업데이트도 검사한다.
- 24-step의 첫 레이어 K projection 토큰 수는 기존 300에서 24로 줄어든다.
- 공유 리트리벌/실험 분기 테스트 86개 통과. 기존 테스트의 `STORM/` 경로는 임시 fixture에서
  `STORM-1/`에 연결했고 실제 workspace에 별칭을 만들지 않았다.

STORM-1 디렉터리에서 실행:

```bash
CUDA_VISIBLE_DEVICES=<유휴_GPU_UUID> python -m unittest discover -s tests -v
python diagnostics/run_shared_retrieval_tests.py
```

## 측정

RTX 3090, PyTorch 2.11.0+cu128, BF16, CPU threads=40.
기준선에도 `VectorizedReplaySampling=True`, `DisableDistributionValidation=True`를 적용했다.
합성 GPU replay 4096개 프레임, 초기화된 모델을 사용했다. 각 구성의 시작 모델·optimizer·scaler·EMA·RNG를
복원하고 준비 호출 4회 후 20회의 중앙값을 측정했다. 두 번째 측정에서는 구성 순서를 뒤집었다.

| 구간 | 기준선 ms, 1차 / 2차 | 최적화 ms, 1차 / 2차 | 시간 감소 |
|---|---:|---:|---:|
| Imagination 1024×16 | 101.263 / 101.536 | 89.857 / 90.181 | 11.3% / 11.2% |
| 통계 3종, 16×55 | 0.612 / 0.613 | 0.443 / 0.445 | 27.6% / 27.5% |
| 스칼라 14개 읽기 | 0.138 / 0.139 | 0.114 / 0.114 | 17.5% / 18.1% |
| 합성 학습 cycle | 165.604 / 166.152 | 153.074 / 153.717 | 7.6% / 7.5% |

합성 학습 cycle에는 replay sampling, 월드 모델 업데이트, warmup 통계, imagination,
actor-critic 업데이트가 포함된다. 환경 수집, hash 삽입, 실제 TensorBoard/W&B 쓰기,
평가·checkpoint는 포함되지 않는다. 구간 경계에서 CUDA를 동기화한 측정이며,
**전체 Hero 실행의 속도 개선율이나 GPU-util 상승률을 측정한 결과가 아니다.**
배치 전송만 적용한 구성에서는 학습 cycle의 일관된 개선이 확인되지 않았다.

별도로 캡처한 한 cycle의 CUDA 프로파일에서 `cudaStreamSynchronize` 호출은
기준선 43회 → batched 26회 → 최적화 17회였다. 통계 함수의 boolean gather 제거로
`aten::nonzero`는 7회 → 0회가 되었다. 프로파일 오버헤드는 시간 측정에 포함하지 않았다.
`aten::item`의 전체 호출 수에는 CPU scalar 읽기도 포함되므로 GPU 동기화 수와 같지 않다.

실제 모델 폭 512, 2개 레이어, 32×24 고정 입력의 BF16 출력 비교:
최대 절대 오차 0.044857, 평균 절대 오차 0.001821, 상대 L2 오차 0.003271.
이 비교에서 RNG 상태는 동일했다. 이는 비트 동일성을 주장하는 결과가 아니다.

재현 명령:

```bash
python diagnostics/benchmark_execution_optimizations.py --gpu <유휴_GPU_UUID> --output /tmp/storm-execution-check --repeats 20 --trace
```

원시 시간·환경·오차·프로파일 count: `diagnostics/execution_optimizations_20260930.json`.
CUDA 타임라인과 연산자 표: `/tmp/storm-execution-final-20260930/`.
