# FLASH threshold / hash sensitivity

`hyperparameter_sensitivity.py`는 `wandb_runs_classification.csv`의 과거 개발
실험을 비교합니다. Python 표준 라이브러리만 사용하고, W&B 접속이나 학습 실행은
하지 않습니다. 입력 CSV와 rebuttal/논문 파일을 수정하지 않습니다.

```bash
python STORM-1/results/hyperparameter_sensitivity.py
```

경로는 스크립트 위치를 기준으로 정하므로 다른 디렉터리에서도 실행할 수 있습니다.
기본 출력은 `STORM-1/results/hyperparameter_sensitivity/`입니다.

## 비교와 선택 규칙

- Threshold: hash=10에서 `3.5`를 기준으로 `3.4`, `3.55`를 비교합니다.
  **세 설정 모두 존재하는 동일한 게임·seed 패널**을 사용합니다.
- Hash: threshold=3.5에서 10-bit를 기준으로 9-bit를 비교합니다.
- `finished`이며 유한한 평가값, 유효한 seed와 생성 시각이 있는 run만 사용합니다.
- 고정 warmup과 CSV상 계산된 warmup이 모두 50,000이어야 합니다.
  `target=16`, `batch_size_reduction=retrieved`, `anchor_offset=-2`를 요구합니다.
  11-bit 기록 중 target=1인 anchor-only run은 hash 비교로 사용할 수 없습니다.
- `Logic`, anchor weight, score combination, value signal, neighbor source,
  checkpoint step 등 `MATCH_FIELDS`의 메타데이터가 정확히 일치해야 합니다.
  `N/A`/빈칸은 같은 결측값끼리만 일치시키며, 현재 코드의 기본값을 대입하지 않습니다.
  따라서 결측 설정의 실제 동일성이 검증된 것은 아닙니다.
- 고정 50k warmup에서는 비활성인 dynamic-warmup 설정은 매칭에서 제외합니다.
- 기본 모드는 Git 기록의 `eval.py`, `agents.py`, `replay_buffer.py`,
  `utils.py`가 완전히 같은 run끼리만 비교합니다. 해당 커밋을 읽을 수 없으면 제외합니다.
  `train.py`는 파일 버전을 출력하되 일치를 강제하지 않습니다. 개발 중 비활성
  `half` 분기 추가 등의 변경이 있었으므로 **선택된 run의 train.py 차이는 확인해야 합니다.**
- 같은 설정·게임·seed의 중복 run은 **생성 시각이 가장 늦은 run**을 선택하고,
  동률이면 Run ID 사전순으로 결정합니다. 서로 다른 매칭 조건이 모두 완전한 패널을
  만들면 가장 최근의 완전한 조건을 사용합니다. 점수 크기는 선택에 사용하지 않습니다.
- 기본적으로 게임당 공통 training seed가 2개 이상 필요합니다.
  `--min-seeds 3` 등으로 높일 수 있습니다. Seed가 2개뿐인 CI는 특히 불안정합니다.

기본 조건에서 hash 비교 패널이 비어 있을 수 있습니다. CSV 설정만 맞추는
**탐색적** 분석은 별도 출력 디렉터리에 명시적으로 실행합니다.

```bash
python STORM-1/results/hyperparameter_sensitivity.py \
  --metadata-only \
  --output-dir STORM-1/results/hyperparameter_sensitivity_metadata
```

이 옵션은 평가 코드까지 다른 기록을 포함할 수 있습니다. 그 결과를 한 변수만
바꾼 통제 실험으로 인용하면 안 됩니다. 같은 파일 버전도 실행 당시 dirty 코드나
CSV에 없는 CLI override, 평가 seed/precision/episode 수까지 보장하지는 않습니다.

통계 없이 선택 결과만 검토하려면:

```bash
python STORM-1/results/hyperparameter_sensitivity.py \
  --audit-only --output-dir /tmp/flash-sensitivity-audit
```

## 지표와 해석

HNS는 기존 `convert_csv_to_excel.py`의 `REFERENCE_SCORES`에 있는 Random/Human
점수로 `(return - random) / (human - random)`을 계산합니다. 해당 파일의 상수만
파싱하므로 pandas/Excel 패키지는 필요하지 않습니다. HNS는 clipping하지 않습니다.

`summary.csv`는 Mean HNS, Median HNS, IQM, Optimality Gap과
**variant − reference** 차이의 percentile bootstrap CI를 저장합니다.
기본값은 50,000회, 95% CI, bootstrap seed=2027입니다.

- 게임은 고정하고, 각 게임 안에서 동일한 training-seed 인덱스를 모든 설정에
  함께 적용하여 재표집합니다. 평가 episode를 독립 training run으로 세지 않습니다.
- Mean/Median/Gap은 게임별 평균 HNS를 사용합니다. Mean과 Gap에서는 게임을
  같은 비중으로 반영합니다. Gap은 `mean(max(0, 1 - game_mean_hns))`입니다.
- IQM은 저장소의 기존 계산과 같이 seed HNS 전체를 모아 양쪽에서
  `floor(N/4)`개씩 제거합니다. Seed가 많은 게임은 IQM에서 비중이 커집니다.
- CI는 각 지표별 구간이며 다중비교 보정을 하지 않습니다. 선택한 개발 설정,
  게임, seed에 조건부인 분석이므로 다른 게임이나 넓은 탐색 범위로 일반화하지 않습니다.
- 평균에서 게임별 차이가 상쇄되는지 확인하도록 절대 게임별 HNS 차이의 평균/최대와
  `per_game.csv`를 함께 저장합니다.

**CI에 0이 포함된다는 사실만으로 성능이 비슷하다고 결론 내리지 않습니다.**
분석 전에 허용 가능한 Mean HNS 차이를 정했다면 아래와 같이 지정할 수 있습니다.
아래 `0.05`는 사용 예시이며 권장 기준이나 자동 기본값이 아닙니다.

```bash
python STORM-1/results/hyperparameter_sensitivity.py \
  --equivalence-margin-hns 0.05
```

`ci_within_equivalence_margin`은 **Mean HNS 차이의 CI 전체가 (-margin, +margin)에
들어가는지**를 나타냅니다. 지정하지 않으면 평가하지 않습니다. 결과를 본 다음
고른 margin은 사후적 기술 기준이지 사전에 정한 동등성 검정이 아닙니다.

## 출력

| 파일 | 내용 |
|---|---|
| `report.md` | Mean HNS 비교와 빈 패널 안내 |
| `summary.csv` | 네 가지 aggregate 지표, 차이 CI, 허용 범위 충족 여부 |
| `per_game.csv` | 게임별 return/HNS, seed 수, 표준편차, paired 차이와 CI |
| `selected_runs.csv` | 선택한 원본 CSV 행, Run ID, 커밋, 다섯 소스 파일 버전 |
| `selection_audit.csv` | 비교별 모든 입력 run의 선택/제외 사유 |
| `metadata.json` | 인자, 입력/스크립트 SHA256, 참조 점수, 선택 범위와 해석 한계 |

`--audit-only`는 마지막 세 파일만 생성합니다. 같은 출력 디렉터리를 다시 사용하면
해당 모드가 생성하는 파일을 덮어쓰므로, 분석 조건별 디렉터리를 지정하는 것이 좋습니다.

검증:

```bash
cd STORM-1
python -m unittest test_hyperparameter_sensitivity
```
