# TD-error-only 가상 시드 보충 시뮬레이션

RNG=2027, 반복=10,000, 게임별 슬롯=3.

기존 Full FLASH–TD-error-only 공통 시드 결과는 고정합니다. 부족한 슬롯마다 같은 게임의 Value Difference vs. Absolute Value 표 입력에서 시드 쌍을 균등 복원추출합니다. Full FLASH 값은 Full FLASH로, Absolute value 값은 가상의 TD-error-only 값으로 사용합니다. 같은 공여 시드를 여러 번 뽑거나 기존 시드와 중복해 뽑을 수 있습니다. 슬롯 수가 3이어도 독립적인 실제 학습 시드 3개를 확보한 것은 아닙니다.

구간은 위 가정에서 누락 슬롯을 채우는 무작위성의 2.5–97.5 백분위수입니다. 실제 TD-error-only 성능의 신뢰구간이나 실제 학습 결과가 아닙니다. 기존 관측값은 재표집하지 않으며 관측값과 공여값의 불확실성을 추정하지 않습니다.

관측 TD 결과가 전혀 없는 게임: Krull, Seaquest, UpNDown.

Mean/Median/Optimality Gap/#Superhuman은 반올림 전 게임별 평균 HNS를 사용합니다. IQM은 반올림 전 시드 슬롯별 HNS를 모아 양끝 floor(N/4)개씩 제거합니다. 표시는 소수 셋째 자리까지이며, 기존 표의 표시값 반올림과 경계에서 차이가 날 수 있습니다.

기존 23게임 (보충 전):

| Metric | Full FLASH | TD-error only |
|---|---:|---:|
| #Superhuman | 7.000 | 7.000 |
| Mean | 1.046 | 0.877 |
| Median | 0.542 | 0.414 |
| IQM | 0.599 | 0.570 |
| Optimality Gap | 0.447 | 0.458 |

completed_observed_games: 23게임, 각 3슬롯

| Metric | FLASH simulation mean [2.5%, 97.5%] | TD/proxy simulation mean [2.5%, 97.5%] | P(FLASH better) |
|---|---:|---:|---:|
| #Superhuman | 7.251 [7.000, 8.000] | 5.745 [5.000, 6.000] | 100.0% |
| Mean | 1.079 [1.056, 1.101] | 0.942 [0.918, 0.965] | 100.0% |
| Median | 0.457 [0.405, 0.604] | 0.432 [0.413, 0.487] | 24.6% |
| IQM | 0.601 [0.575, 0.626] | 0.576 [0.541, 0.610] | 99.8% |
| Optimality Gap | 0.438 [0.423, 0.453] | 0.447 [0.429, 0.465] | 94.8% |

completed_all_games: 26게임, 각 3슬롯

| Metric | FLASH simulation mean [2.5%, 97.5%] | TD/proxy simulation mean [2.5%, 97.5%] | P(FLASH better) |
|---|---:|---:|---:|
| #Superhuman | 8.251 [8.000, 9.000] | 6.745 [6.000, 7.000] | 100.0% |
| Mean | 1.163 [1.143, 1.183] | 1.057 [1.035, 1.079] | 100.0% |
| Median | 0.562 [0.536, 0.635] | 0.461 [0.452, 0.489] | 100.0% |
| IQM | 0.612 [0.589, 0.635] | 0.571 [0.540, 0.602] | 100.0% |
| Optimality Gap | 0.438 [0.425, 0.452] | 0.453 [0.437, 0.469] | 100.0% |

- `example_slots.csv`: 첫 반복의 슬롯별 관측/합성 구분·출처 시드·원점수
- `example_games.csv`: 첫 반복의 게임별 평균·실제 시드 수·추가 슬롯 수
- `observed_metrics.csv`: 보충 전 관측 게임만의 지표
- `simulation_summary.csv`: 각 지표·방법·차이의 평균/백분위수/우세 빈도
- `trials.csv`: 모든 반복의 지표
- `inputs.json`: 입력 경로·SHA256·전체 원점수·설정
