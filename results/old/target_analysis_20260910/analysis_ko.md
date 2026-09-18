분석 기준일: 2026-09-10. 입력은 `STORM/target_comparison.xlsx`, 이 파일의 Readme에 명시된 `STORM/wandb_runs_classification.csv`, `iclr2027_conference.tex`이다. `STORM/results/wandb_runs_classification.csv`는 더 오래된 728행 파일이므로 입력으로 사용하지 않았다. 현재 입력 CSV는 851행이고, 필터와 최신 run 선택 후 Excel에 들어간 420개 run을 모두 재현했다.

**논문에서 retrieval의 추가 가치를 검증할 다음 후보로는 `target=12, anchor_weight=0.12`를 권한다. 다만 현재 결과는 최적 설정이나 PER-only 대비 의미 있는 우위를 확정하지 못한다.** `target=4, anchor 미설정`과 사실상 동률인 선두 후보이며, baseline 대비 더 넓은 게임 범위의 증거만을 중시하면 `target=16, anchor 미설정, hash_bits=10`이 더 설득력 있다. 따라서 “성능이 입증된 최고 config”와 “다음에 검증할 config”를 구분해야 한다.

`target=12, anchor_weight=0.12`는 두 기준과 모두 seed를 맞춘 비교에서 IQM 개선폭이 상대적으로 크다. 이 선택은 논문의 중심 질문인 **anchor-only를 넘어서는 추가 가치**에 맞춘 실험 우선순위다. 전체 게임 평균에서도 target=4보다 좋다는 근거는 확인되지 않았다. 7개 게임 결과에 근거한 선택이므로 26개 게임에서 성공할 확률이나 예상 개선폭을 수치로 보장할 수 없다.

**표를 읽을 때 필요한 구분**

- `Base Summary`와 `Anchor Summary`는 모든 config가 존재하는 **7개 게임**만 포함한다. 게임은 Alien, Amidar, BattleZone, Frostbite, Krull, Qbert, UpNDown이다.
- `Base Scores`/`Anchor Scores`의 점수는 `기준 평균 + 공통 seed에서의 점수 차이`로 보정한 값이다. 기준과 config마다 다른 seed를 사용할 수 있어, 이 값을 실제 학습 run의 평균이나 통계 검정의 관측치로 취급하면 안 된다.
- `Dual Summary`는 후보·baseline·PER-only의 공통 seed만 사용하므로 한 후보를 두 기준과 비교하기에 적합하다. 그러나 **후보마다 게임과 seed 집합이 달라** 후보 간 절대 IQM의 크기로 순위를 매기면 안 된다.
- “모든 config의 동일 관측치”로 다시 제한하면 7개 게임, 총 22개의 game–seed 조합이다. 게임별 개수는 2, 1, 5, 5, 2, 4, 3개이다. Amidar는 1개라 seed 변동성을 추정할 수 없다.
- 원본의 `IQM`은 **게임별 seed 평균 HNS를 정렬한 뒤 양끝 floor(G/4)개를 제거한 평균**이다. G=7에서는 가운데 5개가 남는다. 이 분석의 `IQM_game_workbook`은 비교 가능성을 위해 정확히 이 정의를 재현했다.

일반적인 RL 평가에서 쓰는 rliable의 IQM은 게임 평균을 먼저 내지 않고 **run×game 점수 전체**를 대상으로 한다. 원본 IQM과 같은 지표가 아니다. 이에 게임별 표본 수가 다른 현재 데이터에는 각 게임에 같은 총 가중치를 주고, 각 게임의 run에 그 가중치를 균등 배분한 뒤 중앙 50%의 질량을 평균하는 `IQM_run_equal_game`도 계산했다. 이는 불균등 표본 수에 대한 보조 분석이며 rliable 함수를 그대로 호출한 값은 아니다. 26개 게임×동일한 5개 학습 seed를 갖추면 표준 rliable IQM을 직접 계산하는 것이 좋다. [rliable 공식 지표 구현](https://github.com/google-research/rliable/blob/master/rliable/metrics.py)

**여섯 후보의 직접 비교**

아래는 Excel `Dual Summary`와 재계산이 일치하는 값이다. IQM은 원본 정의이며, 퍼센트는 상대 증가율이다. 승/패는 게임별 평균 점수가 기준보다 높은지로 센다. 각 행의 두 기준 비교만 동일 게임·seed를 사용한다.

| 후보 | 게임 / game–seed 수 | baseline 대비 IQM | PER-only 대비 IQM | baseline 승/패 | PER-only 승/패 |
|---|---:|---:|---:|---:|---:|
| target=4, 미설정 | 12 / 37 | +2.71% | +0.12% | 5 / 7 | 7 / 5 |
| target=8, anchor=0.3 | 8 / 28 | −1.16% | −6.33% | 1 / 7 | 1 / 7 |
| **target=12, anchor=0.12** | **7 / 24** | **+7.33%** | **+2.44%** | **4 / 3** | **3 / 4** |
| target=12, anchor=0.2 | 8 / 28 | −1.08% | −6.25% | 2 / 6 | 3 / 5 |
| target=16, 미설정 | 12 / 33 | −3.03% | −5.47% | 8 / 4 | 4 / 8 |
| target=16, anchor=0.4 | 8 / 27 | +1.29% | −4.01% | 3 / 5 | 3 / 5 |

모든 config에 동일한 22개 관측치만 쓰면 다음과 같다. target=4와 target=12/0.12의 순서는 IQM 정의에 따라 바뀌지만 차이는 매우 작다.

| config | 원본 정의 IQM | run 단위·게임 균등 IQM |
|---|---:|---:|
| baseline | 0.271717 | 0.257316 |
| PER-only | 0.284273 | 0.252029 |
| target=4, 미설정 | **0.288591** | 0.261686 |
| target=12, anchor=0.12 | 0.288297 | **0.262150** |
| target=16, anchor=0.4 | 0.276771 | 0.257795 |
| target=12, anchor=0.2 | 0.267988 | 0.244695 |
| target=8, anchor=0.3 | 0.266715 | 0.238150 |
| target=16, 미설정 | 0.263226 | 0.239159 |

두 선두 후보와 baseline·PER-only만 맞추면 24개 관측치를 사용할 수 있다. 이때 target=12/0.12의 IQM은 0.291051, target=4는 0.289851이다. 차이 +0.001200의 paired bootstrap 95% 구간은 **[−0.024444, +0.026780]**이다. 어느 쪽이 더 좋다고 확정할 수 없다. 원본 보정 요약에서 target=4가 1위라는 사실도 유지해서 해석해야 한다.

**target=12, anchor=0.12의 효과 크기**

7개 게임·24개 공통 game–seed 조합에서의 결과다. 게임별 학습 seed는 2–5개이며 세 방법에 같은 seed를 적용했다.

| 지표 | baseline | PER-only | target=12/0.12 |
|---|---:|---:|---:|
| 원본 정의 IQM | 0.271173 | 0.284111 | 0.291051 |
| Mean HNS | 0.703678 | 0.732459 | 0.669385 |
| Optimality Gap, 작을수록 좋음 | 0.650354 | 0.639839 | 0.635366 |
| run 단위·게임 균등 IQM | 0.256687 | 0.251626 | 0.266021 |

IQM과 optimality gap은 좋아지지만 Mean HNS는 baseline 대비 **−4.87%**, PER-only 대비 **−8.61%**이다. Krull에서의 손실이 Mean HNS에 크게 반영되고, Krull은 높은 HNS 때문에 이 7개 게임의 원본 IQM에서는 제외된다. 따라서 “전체적으로 일관되게 향상”보다는 “중앙부 성능 개선 가능성과 일부 게임 손실이 함께 관측됨”이 맞다. Mean HNS만을 근거로 config를 고르거나, 반대로 Mean 감소를 숨기고 IQM만 제시하지 않는 것이 좋다.

| 게임 | 공통 학습 seed 수 | baseline | PER-only | target=12/0.12 | PER-only 대비 |
|---|---:|---:|---:|---:|---:|
| Alien | 3 | 1179.5 | 1162.2 | 1140.5 | −1.9% |
| Amidar | 2 | 162.9 | 178.2 | 172.4 | −3.3% |
| BattleZone | 5 | 10430.0 | 8750.0 | 10040.0 | +14.7% |
| Frostbite | 5 | 1781.0 | 2106.7 | 1973.9 | −6.3% |
| Krull | 2 | 5311.0 | 5447.5 | 4942.8 | −9.3% |
| Qbert | 4 | 2785.0 | 2768.4 | 3089.1 | +11.6% |
| UpNDown | 3 | 4853.8 | 5304.7 | 5391.5 | +1.6% |

논문에서 예시로 드는 Alien/Frostbite/Qbert 중, 현재 후보는 **Qbert에서 두 기준을 모두 넘지만 Alien에서는 둘 다 못 넘고, Frostbite에서는 baseline만 넘는다.** PER-only 대비 이득은 BattleZone과 Qbert에 집중되어 있다. BattleZone을 제외하고 원본 IQM을 다시 계산하면 PER-only 대비 차이가 +0.006940에서 −0.000585로 바뀐다. 게임 제외 후 IQM에 남는 게임도 바뀌므로 이는 기여도의 정확한 분해가 아니라 게임 구성 민감도 점검이다.

**통계적으로 의미 있는 향상인가**

동일 게임 안에서 학습 seed를 복원 추출하고, 세 방법에 같은 추출 인덱스를 적용하는 paired stratified bootstrap을 50,000회 수행했다. 게임 집합은 고정했다. 아래 구간은 IQM **절대 차이(HNS 단위)**의 percentile 95% 구간이다. 여러 config를 탐색한 효과를 보정하지 않은 탐색적 구간이며 26개 게임에 대한 구간이 아니다.

| target=12/0.12 비교 | 점추정 차이 | 95% 구간 |
|---|---:|---:|
| 원본 IQM, baseline 대비, 24개 삼자 공통 관측치 | +0.019877 | [+0.000496, +0.040509] |
| 원본 IQM, PER-only 대비, 같은 24개 | +0.006940 | **[−0.019183, +0.030459]** |
| 원본 IQM, baseline 대비, 사용 가능한 29개 직접 공통 관측치 | +0.011822 | [−0.010096, +0.033293] |
| run 단위·게임 균등 IQM, baseline 대비, 삼자 공통 24개 | +0.009334 | [−0.009277, +0.032645] |
| run 단위·게임 균등 IQM, PER-only 대비, 같은 24개 | +0.014395 | [−0.008393, +0.036358] |

baseline 대비 첫 구간만 0을 근소하게 제외한다. 그러나 PER-only가 있는 seed로 제한하지 않고 baseline과 비교 가능한 데이터를 더 쓰면 0을 포함하며, run 단위 IQM에서도 0을 포함한다. 모든 config의 동일 22개 관측치를 쓰는 분석에서도 baseline 대비 구간은 [−0.001973, +0.036773]이다. **baseline 대비 향상 가능성은 있으나 견고한 유의성 주장으로 보기 어렵고, PER-only를 넘는 추가 효과는 입증되지 않았다.** 이는 두 방법이 동등하다는 검정 결과도 아니다.

paired 분석은 동일 seed를 실험 블록으로 취급한다. seed가 같아도 retrieval과 코드 변경으로 RNG 소비가 달라지므로 같은 학습 궤적을 의미하지는 않는다. pairing 의존도를 확인하기 위해 같은 seed 집합에서 방법별 독립 bootstrap도 수행했다. target=12/0.12의 원본 IQM 구간은 baseline 대비 [−0.055929, +0.094719], PER-only 대비 [−0.073997, +0.087543]으로 넓어졌다. 어느 분석도 여러 config 탐색, 버전 차이, 평가 프로토콜 오류를 해결하지 못한다.

평가 episode 20개는 하나의 학습된 모델을 반복 평가한 것이다. 5개 학습 seed×20 episode를 **독립 학습 표본 100개로 간주하면 안 된다.** 게임별 점수는 이 100개 episode의 평균으로 표시할 수 있지만, 학습 안정성에 대한 bootstrap의 기본 표본 단위는 5개 학습 run이어야 한다. 소수 run의 점추정만으로 성능을 주장하기 어렵다는 점은 Atari 100k를 다룬 [Agarwal et al.의 원 논문](https://arxiv.org/abs/2108.13264)과도 부합한다.

**baseline 대비 넓은 범위의 결과를 중시할 경우**

`target=16, anchor 미설정`은 baseline과 비교 가능한 20개 게임·59개 관측치에서 원본 IQM이 **0.429878 → 0.451999 (+5.15%)**, 14승 6패이다. IQM 차이 구간도 [+0.007061, +0.035354]이다. run 단위·게임 균등 IQM 차이 역시 +0.017528, 구간 [+0.004820, +0.034249]로 양수다. baseline 개선에 관해서는 target=12/0.12보다 넓은 범위의 근거다.

이 열은 hash_bits=9와 10이 섞여 있다. hash_bits=10인 관측치만으로 제한해도 19개 게임·43개 관측치에서 원본 IQM은 **0.474401 → 0.502657 (+5.96%)**, 구간 [+0.007344, +0.048228]이다. hash_bits=9는 9개 게임·16개 관측치에서 −3.37%이다. 두 hash 설정의 게임·seed가 다르므로 이를 hash_bits 자체의 인과적 효과로 해석할 수는 없지만, 미래 실험을 위해 **10으로 명시할 이유**는 있다.

다만 이러한 구간은 관측 게임 고정·seed pairing·소수 seed에 조건부이다. singleton 게임은 bootstrap에서 점수가 변하지 않는다. 20개 게임 분석에는 BankHeist와 Pong이 각 1개 seed이고, Pong은 5.7→9.8이라는 큰 변화가 있다. 이 두 게임을 제외한 18개 게임·57개 관측치에서는 원본 IQM 증가가 +2.03%, 구간은 [−0.016769, +0.033816]으로 바뀌고, run 단위·게임 균등 IQM 변화는 −0.09%이다. 이 분석은 게임 구성도 바꾸므로 원래 benchmark 결과를 대체하지는 않지만, 적은 seed의 게임에 대한 추가 검증 필요성을 보여준다. 또한 PER-only가 있는 12개 게임으로 제한하면 target=16의 IQM은 baseline 대비 −3.03%, PER-only 대비 −5.47%이다. **26개 게임에서 평균적으로 이길 후보와 논문에서 retrieval의 추가 가치를 보여줄 후보가 현재 데이터에서는 일치하지 않는다.**

반면 target=4는 baseline과 비교 가능한 21개 게임·62개 관측치에서 원본 IQM −0.53%, Mean HNS −1.82%, 8승 13패이다. target=4를 작은 공통 표의 IQM 1위만 보고 26개 게임의 최적 설정으로 결정할 근거는 약하다. target=16과 target=4를 baseline까지 같은 seed로 맞춘 20개 게임·50개 관측치에서는 target=16이 IQM +2.01% 높지만 차이 구간은 [−0.003173, +0.021560]이다.

**config 이름에서 숨겨진 차이와 논문 일치 여부**

1. `target=12, anchor=0.12`는 **global_rebuild_threshold=0.12**이다. target=12/0.2를 포함한 다른 주요 실험은 **0.06**이었다. STORM의 해당 commit YAML과 로컬에서 찾은 선택 run의 W&B config 171개로 점검했다. 이 후보의 로컬 config 13개에서도 0.12가 확인되었다. CSV는 이 필드를 저장하지 않으므로 weight 효과와 rebuild 효과를 분리할 수 없다. 앞으로 config 식별자에 이 값과 실제 로드한 retrieval 모듈의 버전을 포함해야 한다.
2. “anchor 미설정”인 과거 구현은 **실제 유효 그룹 크기 L에 대해 모든 항목의 가중치가 1/L**이었다. 현재 공용 `retrieval.py`는 가중치를 생략하면 **anchor=0.5**가 기본값이다. 현재 YAML을 쓰면서 target만 4 또는 16으로 바꾸면 **anchor=0.12가 그대로 남는다.** 둘 다 과거의 미설정 실험과 다르다. 고정 anchor=1/target도 L<target이면 과거 uniform과 달라진다. 이들을 재현하려면 실제 그룹별 uniform weighting을 명시적으로 구현하거나 과거 구현을 정확히 고정해야 한다.
3. PER-only의 기록상 hash_bits는 11, 대부분의 retrieval 후보는 10이다. target=1이면 이웃을 선택하지 않으므로 보통 hash 폭이 이웃 의미론에 작용하지 않지만, 기존 실행이 다른 config/버전이라는 사실은 남는다. 새로운 ablation에서는 target=1만 바꾸고 나머지 조건과 코드 버전을 맞추는 것이 낫다. 이 방식은 FLASH의 surprise 기반 anchor-only ablation이지 표준 TD-error PER 논문 알고리즘 전체와 같다는 뜻은 아니다.
4. 논문의 방법은 각 그룹 구성원에 동일한 1/L을 준다고 서술한다. target=12/0.12의 현재 구현은 유효 anchor와 L−1개 이웃이 있으면 anchor에 0.12, 각 이웃에 0.88/(L−1)을 준다. L=12이면 각 이웃은 0.08이고 그룹 합은 여전히 1이다. 이 후보를 최종 사용하면 방법 설명에 이 가중치 분배를 반영해야 한다. anchor가 유효하지 않으면 남은 항목은 uniform, 항목이 하나면 가중치 1이다.
5. target은 **anchor를 포함한 최대 그룹 크기**다. target=12는 anchor 1개+이웃 최대 11개이다. max_anchors=16이면 최대 192개 context를 retrieval로 채운다. 모두 유효한 최대 상황에서 총 physical batch는 1024, random은 832, 유효 가중치 합은 832+16=848이다. 실제 그룹 크기와 유효성에 따라 이 수는 달라진다. PER-only는 최대 16개 anchor가 들어가고 가중치 합은 1024다. 따라서 PER-only 대비 차이는 이웃 내용뿐 아니라 batch의 구성과 상대적 가중치 변화도 포함한다.
6. 공용 retrieval 구현은 STORM repository 밖의 `../retrieval.py`에서 로드된다. W&B의 STORM commit만 저장해서는 실행한 retrieval 구현 전체를 고정하지 못한다. 이후 실험에서는 STORM commit, 공용 모듈 commit 또는 파일 SHA-256, resolved config를 함께 기록하는 것이 필요하다.

**26개 게임에 적용할 구체적 제안**

논문의 목적에 맞춰 한 설정만 확장한다면 다음을 사용하고, PER-only와 baseline을 같은 프로토콜로 비교할 것을 권한다. 완성된 신규 실험용 설정은 [proposed_flash_100k.yaml](proposed_flash_100k.yaml), [proposed_per_only_100k.yaml](proposed_per_only_100k.yaml), [proposed_baseline_100k.yaml](proposed_baseline_100k.yaml)에 저장했다. 기존 학습 설정·queue·TeX는 수정하지 않았다.

```yaml
JointTrainAgent:
  SampleMaxSteps: 100000
  ImagineBatchSize: 1024
  ImagineContextLength: 8
  EvalMode: final_only
  EvalEpisodes: 20
  Retrieval:
    enable: true
    target: 12
    anchor_weight: 0.12
    hash_bits: 10
    hash_sample_mode: probs
    use_pca: true
    batch_size_reduction: retrieved
    trigger_mode: z_score
    z_score_threshold: 3.5
    anchor_offset: -2
    warmup_steps: 50000
    ema_alpha: 0.01
    max_anchors: 16
    multiplier: 8
    max_contexts: 256
    global_rebuild_enable: true
    global_rebuild_threshold: 0.12
    global_rebuild_cooldown: 2000
```

기존 결과 재현과 논문용 신규 실험의 budget은 구분해야 한다. 확인한 run config와 현재 기본 설정은 **SampleMaxSteps=102000**이고, training loop는 초기 buffer 수집을 이 한도에 포함한다. 논문은 총 100000 steps라고 쓴다. 위 제안 파일은 **새로운 세 방법의 실험을 모두 100000으로 맞춘 것**이다. 따라서 기존 102000-step 결과의 정확한 재실행 설정이라고 부를 수 없다. 과거 결과를 계속 사용할 계획이면 102000을 명시하거나 동일한 100000-step checkpoint를 확보해야 한다.

학습 seed는 결과 확인 전에 동일한 5개를 고정한다. 예를 들어 현재 선택 데이터에 없는 7001, 7002, 7003, 7004, 7005를 쓸 수 있다. FLASH만으로는 26×5=130회이고, 새 baseline·PER-only까지 같은 조건으로 모두 수행하면 **26×5×3=390회**이다. 각 모델을 동일하게 명시한 20개 평가 seed에서 평가하고, 게임별 5개 학습 run 평균을 저장한다. 별도의 비용·시간 추정은 runtime 정보가 없어 하지 않았다.

실험 우선순위는 target=12/0.12가 아직 평가되지 않은 19개 게임의 범위 확장에 둔다. 계산 예산 때문에 단계적으로 돌리더라도 최종 seed 수와 지표를 사전에 고정하고, 결과를 보며 유리한 seed/게임만 추가하거나 중단하지 않는다. target=4 또는 target=16을 추가 비교한다면 config 선택용 결과와 최종 검증 결과를 구분해서 기록한다.

현재 `eval.py`는 `reset()`에 명시적 평가 seed를 전달하지 않으므로 “정해진 20개 평가 seed”를 이미 구현했다고 볼 수 없다. 또한 done episode의 return을 `final_rewards`에 추가한 **뒤** `sum_reward += reward`를 수행해 종료 transition의 reward가 해당 episode return에서 빠지고 다음 누적에 들어갈 수 있다. 26개 게임의 큰 실험 전 평가 seed/episode 대응과 terminal reward 누적 순서를 정리해야 한다. 이 분석은 저장된 점수를 그대로 평가했으며 해당 문제의 수치적 영향은 추정하지 않았다.

최종 판단에서는 run×game IQM과 95% 신뢰구간을 주 지표로 고정하고, Mean/Median HNS, optimality gap, 게임별 점수와 학습 seed 편차를 함께 제시한다. retrieval의 구체적 가치를 더 분리하려면 동일 개수·가중치의 random neighbor를 넣는 대조군도 유용하다. 현재 PER-only 비교만으로 hashing을 통한 구조적 유사성이 향상의 원인이라고 결론 낼 수 없다. overhead도 retrieval 후보 수, 유효 context 수, rebuild 횟수와 전체 학습 시간을 실제로 기록해야 논문의 “negligible overhead”를 뒷받침할 수 있다.

**논문 서술에 대한 판단**

현재 본문/appendix의 “removing Retrieval results in a severe performance drop”와 “anchor injection alone does not reproduce the performance of the full method”는 이 결과표를 근거로 유지하기 어렵다. 적절한 표현은 다음과 같다.

> On the seven evaluated games, the selected configuration improved the game-mean trimmed HNS over STORM, while its incremental benefit over anchor-only replay remained uncertain. Effects varied across games, motivating a controlled evaluation on the full Atari 100k suite.

여기서 game-mean trimmed HNS라는 표현은 현재 표의 지표 정의를 드러내기 위한 것이다. 표준 run 단위 IQM으로 변경한 최종 결과에는 그 정의에 맞춰 문장을 다시 써야 한다. Alien/Frostbite/Qbert의 공통 향상이나 대부분 게임에서 일관된 이득도 현재 선택 후보의 결과에 맞춰 수정해야 한다. 논문의 큰 가설은 후속 실험의 동기로 유지할 수 있지만 현재 수치보다 강한 증거로 표현하면 안 된다.

**분석 산출물과 재현**

- [신뢰구간 그림](iqm_intervals.png), [PDF](iqm_intervals.pdf): 후보별 삼자 공통 비교. 서로 다른 후보 행은 게임 집합이 다를 수 있다.
- [전체 bootstrap 비교](bootstrap_comparisons.csv): 두 IQM 정의, mean, median, gap, 공통 패널, pairing 민감도 및 hash 폭 분석.
- [모든 config의 동일 관측치](identical_panel_scores.csv), [동일 관측치 지표](identical_panel_metrics.csv).
- [게임별 직접 비교](game_comparisons.csv), [게임별 seed 수](coverage.csv), [선택 run](selected_runs.csv).
- [로컬 W&B config 점검](local_config_audit.csv), [입력 checksum 및 분석 조건](metadata.json).
- [분석 코드](analyze.py): 아래 명령은 입력을 수정하지 않고 이 폴더에 수치와 그림을 다시 생성한다.

```bash
/home/ai2lab/miniconda3/envs/storm/bin/python \
  /media/storage_data/ai2lab/choemj/STORM/results/target_analysis_20260910/analyze.py
```

참조한 구현 위치: `compare_target_results.py`의 `load_latest`, `compare`, `summarize`, `dual_comparisons`; `retrieval.py`의 `RetrievalContextManager.__init__`와 `retrieve_contexts`; `STORM/train.py`의 batch 구성 및 최종 평가; `STORM/eval.py`의 `eval_episodes`. 기존 변환기의 baseline 과거 데이터 보존 및 Frostbite 수동 복구 2건도 원본 선택 로직대로 유지했다. 그 2건의 원 실험까지 별도로 재검증한 것은 아니다.
