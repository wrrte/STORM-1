# Frostbite critic 정렬 점수 평가

`probing/eval_frostbite_value.py`는 단일 체크포인트 평가와 두 체크포인트의 공통 문맥 비교를 지원합니다.
MLP나 critic을 학습하지 않으며 체크포인트를 덮어쓰지 않습니다.

## 단일 모델 평가

Baseline 모델 없이 TD-only의 두 정렬 점수를 계산할 수 있습니다.
아래 명령은 논문에서 사용한 관측·행동 데이터를 재사용하며, TD-only 체크포인트만 로드합니다.

```bash
conda activate storm
cd /home/choemj/STORM-1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python probing/eval_frostbite_value.py \
  --run Frostbite-td_only-b3f54884d761-seed710-attempt1_O \
  --config config_files/STORM.td_error_only.yaml \
  --step 100000 \
  --seed 2027 \
  --data results/old/seed_3710/shared_contexts.npz \
  --bootstrap 10000 \
  --bootstrap-unit episode \
  --output results/frostbite_value_td_only_seed710_single
```

1. **문맥별 상관의 평균:** 각 문맥에서 17단계와 예측 가치 사이의 Spearman ρ / Kendall τ_b를 구한 뒤 문맥 전체에서 평균합니다.
2. **전역 평균 프로파일의 상관:** 각 단계의 예측 가치를 모든 문맥에서 평균한 뒤 17단계와 상관을 계산합니다.

단일 모델 모드에서는 두 방식 모두 에피소드 bootstrap 95% CI를 계산합니다.
전역 프로파일의 CI는 각 재표집에서 프로파일을 다시 평균하고 상관을 다시 계산합니다.
재표집된 프로파일이 상수이면 상관이 정의되지 않으므로 해당 표본은 CI에서 제외하고 횟수를 기록합니다.

두 정렬값과 CI는 `report_ko.md`에 함께 표시됩니다. `statistics.json`에는 다음 항목이 있습니다.

- `metrics.spearman` / `metrics.kendall`: 문맥별 평균(`mean`), CI(`ci95`), 정의되지 않는 문맥 수
- `global_profile_correlations`: 전역 프로파일 상관
- `global_profile_ci95`: 전역 프로파일 상관의 CI
- `global_profile_undefined_bootstrap_repetitions`: 상수 프로파일로 제외된 bootstrap 표본 수

원자료는 `model_values.npz`에 저장됩니다. CSV의 값 열은 `value`이고 모델 간 차이는 계산하지 않습니다.
`--data`를 생략하면 평가 대상 모델의 정책으로 새 데이터를 수집하므로 평가 문맥의 분포가 달라집니다.
학습 방법별 정렬 점수를 같은 조건에서 비교하려면 동일한 `--data`를 사용하세요.

## 두 모델 비교

STORM 디렉터리에서 다음 명령의 두 run 이름과 이미지 폴더를 실제 경로로 바꾸세요.
두 run은 동일한 모델 구조/config를 사용해야 합니다. `--config`로 해당 YAML을 지정할 수 있습니다.

```bash
conda activate storm
cd /home/choemj/STORM-1
export CUDA_VISIBLE_DEVICES=5
PYTHONPATH=. python probing/eval_frostbite_value.py \
  --baseline-run 'BASELINE_RUN_NAME' \
  --flash-run 'FLASH_RUN_NAME' \
  --step 100000 \
  --template-dir frostbite_igloo_templates \
  --frames 20000 \
  --output results/frostbite_value_comparison
```

각 run의 `ckpt/<run>/world_model_100000.pth`와 `agent_100000.pth`,
`stage_00.png`~`stage_16.png` 이글루 패치(높이 7, 너비 16)가 필요합니다.
현재 기본 경로에 없다면 `--template-dir`에 이동한 폴더를 지정하세요.

비교 모드에서는 두 run을 함께 지정합니다. `--run_name`은 `--baseline-run`의 별칭이며
`--flash-run`도 필요합니다. 단일 모델 모드의 `--run`과 비교 모드 인자는 함께 사용할 수 없습니다.

## 평가 절차

1. Baseline 정책으로 관측·행동·RAM을 한 번 수집하고 `shared_contexts.npz`로 저장합니다.
   에피소드 ID, 에피소드 내 프레임 순서, 종료 여부도 기록합니다.
2. 이글루 단계가 유지되는 길이 9~64의 블록을 고릅니다. 에피소드 경계를 넘지 않습니다.
3. 각 블록 전체의 이글루를 17단계로 교체합니다. 앞 8프레임은 문맥 준비에 사용합니다.
   이후 각 평가 시점마다 17개 가치를 얻습니다. 두 모델에 관측·행동·블록이 모두 같습니다.
4. 기존의 확률적 latent sampling을 유지하되 블록별 난수 시드를 모델·단계 간 공유합니다.
   무관한 난수 차이를 줄이기 위한 것으로, 여러 latent 표본에 대한 기대값 추정은 아닙니다.
5. 개별 프레임의 Spearman rho/Kendall tau, 전체 평균 가치 곡선의 상관계수를 계산합니다.
6. 동일 에피소드를 두 모델에서 함께 재표집하는 paired cluster bootstrap을 10,000회 수행합니다.

위 절차는 비교 모드입니다. 단일 모델 모드에서는 `--run` 모델의 수집·평가를 한 번만 수행합니다.
기존 `--data`를 지정하면 두 모드 모두 수집을 생략하고 그 문맥을 재사용합니다.
데이터 수집 정책은 `protocol.json`에 기록합니다. 배경 노이즈를 주입하는 MLP probing과는 별개의 평가입니다.

`--frames`는 수집할 관측 수입니다. 문맥 준비와 짧은 블록 제외 때문에 실제 평가 프레임 수는 더 작습니다.
마지막 에피소드는 수집 예산 때문에 일부만 포함될 수 있으며 완료 에피소드 수는 `protocol.json`에 기록합니다.

## 결과 읽기

먼저 `results/frostbite_value_comparison/report_ko.md`를 여세요.

- Baseline/FLASH의 평균 상관계수와 각 95% CI
- FLASH − Baseline의 평균 차이(절대 효과 크기)와 95% CI
- 각각과 차이의 중앙값, Q1/Q3, FLASH가 더 높은 문맥 비율
- 두 모델의 상관계수 분포 및 짝지어진 차이 분포 그래프

`statistics.json`에는 IQR, 개선/동률/악화 비율, 차이의 CI가 0을 제외하는지,
정의되지 않는 상관계수 개수와 global profile 상관계수도 있습니다.

| 파일 | 내용 |
|---|---|
| `model_values.npz` | 단일 모델의 평가 문맥 × 17단계 가치 및 문맥 ID |
| `paired_values.npz` | 두 모델의 평가 프레임 × 17단계 가치 및 문맥 ID |
| `per_context_correlations.csv` | 프레임별 상관계수, 두 모델 차이, 에피소드/블록 ID |
| `per_stage_values.csv` | 프레임·단계별 두 모델의 예측 가치 |
| `global_profile.csv` | 단계별 전체 평균 가치 |
| `correlation_distributions.png/pdf` | 논문용 분포 그래프 |
| `protocol.json` | run, 체크포인트 시점, 수집 출처와 설정 |
| `used_templates.npz` | 이번 평가에 사용한 실제 이글루 템플릿 |

상수 예측의 상관계수는 0으로 채우지 않습니다. 해당 모델의 상관계수를 정의 불가로 기록하고,
짝지어진 비교에서는 어느 한쪽이라도 정의 불가인 문맥을 제외합니다.
원래 방식대로 모델별 유효 문맥 전체에서 계산한 점 추정치도 JSON의 `*_all_defined`에 별도로 남깁니다.

## 신뢰구간의 의미

기본 `--bootstrap-unit episode`는 연속 프레임을 개별 독립 표본으로 취급하지 않습니다.
에피소드를 복원 추출하고, 선택된 에피소드의 모든 유효 문맥을 포함해 프레임 가중 평균을 다시 계산합니다.
따라서 길이가 다른 에피소드도 처리합니다. CI는 percentile 방식이며 별도 p-value는 산출하지 않습니다.

이 불확실성은 **고정 체크포인트의 평가 문맥**에 관한 것입니다.
학습 시드 간 변동까지 포함하지 않습니다. 에피소드가 적으면 탐색적으로 해석해야 하며,
유효 에피소드가 1개뿐이면 CI를 표시하지 않습니다. `--bootstrap-unit block`은 보조 분석 옵션이며,
같은 에피소드 내 블록 사이의 의존성을 반영하지 못합니다.

기존 비교 모드의 CI는 문맥별 상관 평균과 그 차이에 대한 것입니다.
단일 모델 모드는 여기에 더해 전역 평균 프로파일의 상관에 대한 CI도 계산합니다.

CI가 0을 제외하더라도 절대 rho가 작을 수 있습니다. 이 결과만으로 공간적 얽힘의 원인이나 해소를 입증하지 않습니다.

## 재사용

GPU 추론 없이 저장된 가치로 통계만 다시 계산할 수 있습니다.

```bash
PYTHONPATH=. python probing/eval_frostbite_value.py \
  --analyze-only \
  --output results/frostbite_value_comparison \
  --bootstrap 10000
```

다른 모델 쌍을 같은 문맥으로 비교하려면 `--data results/frostbite_value_comparison/shared_contexts.npz`와
새 `--output`을 함께 지정하세요. 존재하는 `--data`는 재사용하므로 `--frames`에 따라 재수집하지 않습니다.
`--analyze-only`는 `model_values.npz` / `paired_values.npz`를 자동으로 구분합니다.
둘 중 하나가 이미 있는 출력 폴더에서는 새 평가를 중단합니다. 새 평가에는 다른 `--output`을 지정하세요.
