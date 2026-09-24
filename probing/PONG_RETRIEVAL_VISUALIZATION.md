# Pong 체크포인트의 오프라인 retrieval 시각화

`visualize_pong_retrieval.py`는 저장된 모델·replay·EMA 통계·hash bucket으로 분석합니다.
환경 rollout, 학습 재개, 모델 업데이트, EMA 업데이트, hash bucket 재구축을 수행하지 않습니다.
`train.py`를 import하지 않으며, 결과는 별도 output 디렉터리에 저장합니다.

## 실행

```bash
conda activate storm
cd /home/choemj/STORM-1
# matplotlib이 없다면 한 번 설치
python -m pip install matplotlib

python probing/visualize_pong_retrieval.py \
  --checkpoint ckpt/Pong-9999_Shared \
  --output results/pong_retrieval_analysis \
  --device cuda:0
```

GPU 없이 분석하려면 `--device cpu`를 사용합니다. 생략하면 사용 가능한 장치를 선택합니다.
입력은 `Pong-9999_Shared` 또는 그 안의 `shared_warmup_50935` 디렉터리입니다.
출력 디렉터리는 비어 있어야 합니다. 이미 결과가 있으면 새 이름을 사용하세요.

## 선택 방법과 의미

1. 양의 reward를 기록한 transition을 찾아 해당 transition과 이후 3개 transition을 후보로 삼습니다.
   `obs[q] --action[q]--> obs[q+1]`에서 `reward[q] > 0`이면 `obs[q+1]`이 득점 직후 화면입니다.
   `--post-score-steps 0`은 득점 reward가 발생한 transition 자체만 분석합니다.
2. 각 후보 `s`에 대해 기본 64개 관측(`s-62`부터 `s+1`)과 기록된 action을 입력합니다.
   학습 코드에서 critic에 전달하는 posterior latent와 transformer feature를 재계산합니다.
   종료 경계를 건너거나 필요한 과거·다음 관측이 없는 후보는 제외합니다.
3. `|reward[s] + gamma*(1-done[s])*V[s+1] - V[s]|`와 설정된 value signal을 사용합니다.
   현재 체크포인트 설정에서는 `value_diff = V[s]-V[s-1]`이고, 저장된 EMA로 표준화한 뒤
   `relu(z_TD) * softplus(z_value_diff)`가 surprise score입니다. EMA는 분석 내내 고정합니다.
4. 설정의 threshold 이상인 후보를 높은 점수부터 확인합니다. `anchor = s+1+anchor_offset`입니다.
   저장된 `anchor_offset=-3`이면 anchor는 `s-2`입니다. **높은 surprise가 발생한 transition과
   검색 anchor는 다른 시점**이며, 그림과 JSON에 두 위치를 모두 표시합니다.
5. 해당 anchor 하나를 원래 `RetrievalContextManager.retrieve_contexts()`에 넣습니다.
   저장된 bucket에서 무작위 후보를 뽑고 현재 encoder로 hash를 다시 확인한 뒤,
   target/max_contexts 제한까지 적용한 실제 반환 인덱스를 사용합니다.
   neighbor를 하나 이상 반환한 첫 후보를 선택하며, 시도 내역도 저장합니다.
   이 검색은 거리순 k-NN이 아닙니다.

이는 **체크포인트에서 새로 계산한 점수와 새로 실행한 검색 결과**입니다.
warmup 체크포인트에 과거 검색 이력이 없으므로 과거 학습 당시 결과를 재현했다는 의미는 아닙니다.
모델은 eval 모드이고 기본 latent는 학습과 같은 `random_sample` 방식입니다.
seed(기본: 저장된 seed)를 고정하지만 장치·정밀도·batch-size에 따라 결과가 달라질 수 있습니다.
학습 중 배치별 최대 점수 trigger 선택 대신, 득점 주변의 후보를 비교하는 분석 절차입니다.

## 결과 파일

| 파일 | 내용 |
|---|---|
| `neighbors.png`, `neighbors.pdf` | 각 행에 anchor 또는 neighbor의 **관측 / 기록된 action / 다음 관측**. 반환된 순서와 모든 neighbor를 그대로 표시 |
| `point_event.png`, `point_event.pdf` | 득점 주변 연속 화면, action·reward, 득점 직후 화면, surprise·anchor 위치 |
| `surprise_scores.png`, `surprise_scores.pdf` | 선택된 득점 이벤트 주변의 재계산 점수와 threshold |
| `candidate_scores.csv` | 모든 유효 후보의 reward/surprise/anchor 인덱스, 점수, TD error, value, z-score |
| `analysis.json` | 입력 체크포인트, 분석 설정, 고정 EMA, 선택 근거, 실제 반환 인덱스·weight·action·reward |
| `transitions.npz` | 원본 uint8 `obs`, `next_obs`, `next_obs_valid`, action, reward, termination, 인덱스, weight, 반환된 context의 관측·action |

observation은 replay에 저장된 64×64 RGB 화면입니다. 원래 ALE 해상도 화면은 이 체크포인트에
저장되어 있지 않습니다. 모델의 복원 이미지나 상상한 다음 화면은 사용하지 않습니다.
next observation은 같은 환경의 다음 replay 관측입니다. 종료 transition 또는 replay의 마지막
transition에서는 실제 다음 화면을 확정할 수 없어 그림에 unavailable을 표시하고
NPZ의 `next_obs_valid=False`로 기록합니다. 해당 NPZ의 0 배열을 실제 화면으로 해석하면 안 됩니다.
경계 판별은 체크포인트에 저장된 termination을 사용합니다.

## 선택 옵션

```bash
# 한 득점 이벤트를 지정 (candidate_scores.csv의 reward_pointer 참조)
python probing/visualize_pong_retrieval.py \
  --checkpoint ckpt/Pong-9999_Shared \
  --output results/pong_one_point \
  --reward-pointer 12345

# surprise transition 바로 다음 관측을 anchor로 분석
# 저장된 -3 offset의 원래 검색 설정을 의도적으로 바꾸는 분석입니다.
python probing/visualize_pong_retrieval.py \
  --checkpoint ckpt/Pong-9999_Shared \
  --output results/pong_post_surprise_anchor \
  --anchor-offset 0
```

- `--target 16`: anchor 포함 최대 16개 context. 생략하면 저장된 설정을 사용합니다.
- `--multiplier 16`: hash 재확인 전 후보 샘플 수의 배수. 생략하면 저장된 설정을 사용합니다.
- `--min-score 3.5`: 선택 threshold 재정의. 기본값은 저장된 trigger 설정입니다.
- `--latent-mode probs`: categorical sample 대신 확률 벡터를 critic 입력에 사용하는 별도 분석입니다.
  이는 여러 latent sample에서 평균 낸 기대 value가 아닙니다. hash encoding 모드는 저장된 설정을 유지합니다.
- `--action-labels NOOP FIRE RIGHT LEFT RIGHTFIRE LEFTFIRE`: Pong minimal action ID 0~5에 이름 추가.
  기본 그림에는 action ID를 표시합니다.

threshold를 넘으며 neighbor를 반환하는 후보가 없으면 `candidate_scores.csv`와 `analysis.json`을
남기고 종료합니다. 더 낮은 threshold나 더 큰 multiplier를 사용하면 변경된 분석 설정도 JSON에 기록됩니다.

요청에 따라 이 스크립트는 작성만 했으며 실행·검증하지 않았습니다.
