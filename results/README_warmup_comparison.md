# Warmup 비교

```bash
python -m pip install -r STORM-1/results/requirements_warmup.txt
python STORM-1/results/compare_warmup.py
```

필요 패키지: `numpy pandas openpyxl matplotlib`. 그래프가 필요 없으면 `--no-plots`를 사용하면 됩니다.
기본 입력과 출력 경로는 스크립트 위치 기준이므로 실행 디렉터리에 영향을 받지 않습니다.

`Results` 시트에서 아래 **세 Config 문자열과 정확히 일치하는 행만** 읽습니다.
지표를 추가할 때도 이 허용 목록은 그대로 적용됩니다. 과거 실험, baseline, 다른 anchor/offset,
ablation 행과 숨겨진 `Ablation Results` 시트는 사용하지 않습니다.

| 요청 warmup | Config |
| --- | --- |
| 30,000 | `target: 16 (anchor r: 1) [warmup: 30000]` |
| 50,000 | `target: 16 (anchor r: 1)` |
| 70,000 | `target: 16 (anchor r: 1) [warmup: 70000]` |

게임 셀의 Random/Human 참조 점수로 `HNS = (score - Random) / (Human - Random)`을 계산합니다.
HNS 1은 인간 수준이며 백분율로 바꾸거나 음수·초인간 점수를 자르지 않습니다.
`(w: 30332)` 같은 실제 분기 step은 보관하되, 비교 그룹은 요청값인 30,000으로 유지합니다.

한 시드 셀에 여러 결과가 있으면 엑셀에 표시된 순서에서 **첫 번째 유효한 완료 점수**를 선택합니다.
변환기의 최신순·hash bits 우선순위를 이어받고, 중복 run을 추가 시드로 세지 않습니다.
`RUNNING`/`QUEUED` 항목은 점수로 쓰지 않습니다. `RUNNING, 123.00`이면 이전 완료 점수인 123을 사용합니다.
엑셀의 취소선 셀은 기존 제외 시드이므로 집계에서 제외합니다. 원본 엑셀은 수정하지 않습니다.

주 비교는 세 설정 모두에 점수가 있는 **동일한 (게임, 시드)의 교집합**만 사용합니다.
완료 게임이 더 많은 설정이 유리해지는 것을 방지합니다. `--min-seeds 2`를 주면 공통 시드가
게임당 2개 이상인 게임만 남깁니다. 교집합이 없으면 주 지표는 빈 값으로 남기며 다른 표본으로 대체하지 않습니다.

`update_tex.py`의 집계 관례를 따르되, 게임 평균을 중간에 반올림하지 않습니다.
게임별 시드 평균 HNS를 `m_g`, 전체 공통 (게임, 시드) 수를 `N`이라고 할 때:

| 지표 | 계산 | 좋은 방향 |
| --- | --- | --- |
| Mean | 게임별 `m_g`의 평균 | 높음 |
| Median | 게임별 `m_g`의 중앙값 | 높음 |
| IQM | 시드별 HNS를 모두 모아 정렬하고 양끝에서 각각 `floor(N/4)`개를 제거한 평균 | 높음 |
| Optimality Gap | 게임별 `max(0, 1 - m_g)`의 평균 | 낮음 |

IQM은 게임 평균을 잘라 계산하지 않습니다. 게임별 시드 수가 다르면 IQM에서는 시드가 많은 게임이 더 큰 비중을 갖습니다.
Mean, Median, Optimality Gap은 각 게임에 동일한 비중을 둡니다.
Optimality Gap은 시드마다 gap을 계산한 뒤 평균하는 정의와 다를 수 있습니다.

기본 2,000회 bootstrap은 게임 집합을 고정하고 게임 안의 시드를 복원 추출합니다.
세 설정에 같은 시드 추출을 적용하며, 95% percentile CI와 50k 대비 차이의 CI를 제공합니다.
모든 게임의 공통 시드가 하나씩이면 시드 변동성을 추정할 수 없어 CI를 비워 둡니다.
일부 게임만 단일 시드이면 그 게임은 bootstrap에서 고정됩니다.

기본 출력 폴더: `STORM-1/results/warmup_comparison/`.

- `warmup_comparison.xlsx`: 주 비교, 50k 대비 차이, 게임별 HNS, coverage, 선택한 점수, 셀 감사 내역, 계산 정의.
- `summary.csv`, `delta_vs_50000.csv`, `per_game.csv`, `coverage.csv`, `selected_scores.csv`.
- `available_diagnostic.csv`: 세 설정 각각의 완료 점수 전체를 사용한 참고 집계. 표본이 달라 순위 비교용으로 사용하지 않습니다.
- `metrics_vs_warmup.png` / `.pdf`: 네 지표의 warmup별 변화. 공통 표본이 없으면 생성하지 않습니다.
- `hns_by_game.png`: 공통 시드로 계산한 게임별 HNS 비교.

```bash
python STORM-1/results/compare_warmup.py --bootstrap-reps 10000 --seed 42
python STORM-1/results/compare_warmup.py --min-seeds 2 --no-plots
python STORM-1/results/compare_warmup.py --input /path/to/converted_results.xlsx --output-dir /path/to/report
```
