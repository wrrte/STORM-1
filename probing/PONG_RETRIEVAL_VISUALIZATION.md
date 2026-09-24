# Pong 체크포인트의 오프라인 retrieval 시각화

`visualize_pong_retrieval.py`는 저장된 모델·replay·EMA 통계를 읽고,
**PCA 기반 global rebuild를 먼저 수행한 뒤** anchor와 neighbor를 추출합니다.
환경 rollout, 학습 재개, 모델 업데이트, EMA 업데이트는 수행하지 않습니다.
PCA와 hash bucket은 메모리에서 새로 계산하며 원본 체크포인트는 덮어쓰지 않습니다.
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
기본 실행마다 OS 난수로 새 seed를 만들고, 트리거 조건을 통과한 anchor 중 무작위로 선택합니다.
neighbor도 원래 retrieval 함수의 무작위 샘플링으로 추출합니다. 우연히 같은 anchor가 다시 나올 수 있습니다.
분석 seed는 콘솔과 `analysis.json`에 기록하며, `--seed 1234`처럼 지정하면 고정할 수 있습니다.
이는 체크포인트의 학습 seed와 별개입니다. PCA와 latent sampling도 이번 분석 seed를 사용합니다.

위 명령을 그대로 반복해도 됩니다. 지정한 output이 비어 있으면 그곳에 저장하고,
이미 결과가 있으면 `results/pong_retrieval_analysis/run_seed숫자_고유문자열/`에 새로 저장합니다.
콘솔에 출력되는 실제 결과 경로를 이후 `pong_retrieval_events.py --analysis`에 사용하세요.
그림 재배치 도구의 `--output`은 여전히 새 디렉터리를 지정해야 합니다.

## 선택 방법과 의미

현재 `train.py`는 shared warmup 분기 시 `global rebuild → 체크포인트 저장` 순서로 동작합니다.
이 분석은 저장 당시 rebuild 여부와 관계없이 PCA를 새로 계산합니다.
과거 PCA 실행 직전의 난수 상태를 재현하는 절차는 아니므로 기존 projection과 같다는 의미는 아닙니다.

1. 체크포인트를 읽은 직후 원래 `RetrievalContextManager.rebuild_all_hash_buckets()`를 호출합니다.
   `use_pca=True`로 전체 replay를 현재 encoder로 인코딩하고 PCA projection·평균을 다시 계산한 뒤
   전체 hash bucket과 인덱스를 재구축합니다. encoding chunk size는 실제 warmup 저장 경로와
   같은 1024이며, `hash_bits`, `max_pca_samples`, `hash_sample_mode`는 저장된 설정을 사용합니다.
   새 PCA projection을 얻지 못하면 이전 projection으로 진행하지 않고 중단합니다.
2. 양의 reward를 기록한 transition을 찾아 해당 transition과 이후 3개 transition을 후보로 삼습니다.
   `obs[q] --action[q]--> obs[q+1]`에서 `reward[q] > 0`이면 `obs[q+1]`이 득점 직후 화면입니다.
   `--post-score-steps 0`은 득점 reward가 발생한 transition 자체만 분석합니다.
3. 각 후보 `s`에 대해 기본 64개 관측(`s-62`부터 `s+1`)과 기록된 action을 입력합니다.
   학습 코드에서 critic에 전달하는 posterior latent와 transformer feature를 재계산합니다.
   종료 경계를 건너거나 필요한 과거·다음 관측이 없는 후보는 제외합니다.
4. `|reward[s] + gamma*(1-done[s])*V[s+1] - V[s]|`와 설정된 value signal을 사용합니다.
   현재 체크포인트 설정에서는 `value_diff = V[s]-V[s-1]`이고, 저장된 EMA로 표준화한 뒤
   `relu(z_TD) * softplus(z_value_diff)`가 surprise score입니다. EMA는 분석 내내 고정합니다.
5. 학습 코드의 `add_batch_transitions()`와 같이 초기 `skip_len` 이후 모든 유효 transition의
   점수를 계산합니다. 후보 `s`가 **해당 시퀀스의 argmax이며 저장된 trigger threshold 이상**인
   경우에만 실제 trigger 후보로 인정합니다. 동점일 때도 원래 `torch.max`처럼 먼저 나온 시점을 택합니다.
   그 조건을 통과한 서로 다른 anchor들을 무작위 순서로 확인합니다. `anchor = s+1+anchor_offset`입니다.
   저장된 `anchor_offset=-3`이면 anchor는 `s-2`입니다. **높은 surprise가 발생한 transition과
   검색 anchor는 다른 시점**이며, 그림과 JSON에 두 위치를 모두 표시합니다.
6. rebuild 이후의 인덱스에서 anchor의 hash key를 얻고,
   해당 anchor 하나를 원래 `RetrievalContextManager.retrieve_contexts()`에 넣습니다.
   새 PCA bucket에서 무작위 후보를 뽑고 현재 encoder로 hash를 다시 확인한 뒤,
   target/max_contexts 제한까지 적용한 실제 반환 인덱스를 사용합니다.
   neighbor를 하나 이상 반환한 첫 후보를 선택하며, 시도 내역도 저장합니다.
   점수가 높은 순으로 선택하지 않습니다. 이 검색은 거리순 k-NN이 아닙니다.

이는 **체크포인트에서 새로 계산한 점수와 새로 실행한 검색 결과**입니다.
warmup 체크포인트에 과거 검색 이력이 없으므로 과거 학습 당시 결과를 재현했다는 의미는 아닙니다.
모델은 eval 모드이고 기본 latent는 학습과 같은 `random_sample` 방식입니다.
`--seed`를 고정하더라도 장치·정밀도·batch-size에 따라 결과가 달라질 수 있습니다.
각 분석 시퀀스에 학습과 같은 argmax·threshold 조건을 적용하지만, 과거 학습 배치·EMA 이력을
재생하는 것은 아닙니다. 득점 주변의 조건을 만족한 anchor들에서 새로 무작위 표본을 뽑는 분석입니다.

## 결과 파일

| 파일 | 내용 |
|---|---|
| `neighbors.png`, `neighbors.pdf` | 각 행에 anchor 또는 neighbor의 **관측 / 기록된 action / 다음 관측**. 반환된 순서와 모든 neighbor를 그대로 표시 |
| `point_event.png`, `point_event.pdf` | 득점 주변 연속 화면, action·reward, 득점 직후 화면, surprise·anchor 위치 |
| `events/anchor/point_event.png`, `.pdf` | anchor를 t=0으로 한 t−2부터 t+6까지의 실제 연속 화면 |
| `events/neighbor_01/point_event.png`, `.pdf` 등 | 각 neighbor의 t−2부터 t+6까지 실제 연속 화면. 번호는 검색 반환 순서이며 재정렬하지 않음 |
| `all_point_events.png`, `.pdf` | anchor와 모든 neighbor를 행으로, 동일한 상대 시점을 열로 배치한 전체 비교표 |
| `point_events.npz`, `point_events.json` | 재배치에 사용할 실제 연속 프레임·action·reward·인덱스·유효성 mask와 메타데이터 |
| `surprise_scores.png`, `surprise_scores.pdf` | 선택된 득점 이벤트 주변의 재계산 점수와 threshold |
| `candidate_scores.csv` | 모든 후보의 인덱스·점수·TD error·value·z-score와 `trigger_passed`, 시퀀스 최대 점수·위치. CSV만 점수순 정렬이며 선택은 무작위 |
| `analysis.json` | 분석 seed, 입력 체크포인트, PCA rebuild 정보, 고정 EMA, trigger 조건, 무작위 선택 후보 수·시도 내역, 실제 반환 인덱스·weight·action·reward |
| `transitions.npz` | 원본 uint8 `obs`, `next_obs`, `next_obs_valid`, action, reward, termination, 인덱스, weight, 반환된 context의 관측·action |

observation은 replay에 저장된 64×64 RGB 화면입니다. 원래 ALE 해상도 화면은 이 체크포인트에
저장되어 있지 않습니다. 모델의 복원 이미지나 상상한 다음 화면은 사용하지 않습니다.
next observation은 같은 환경의 다음 replay 관측입니다. 종료 transition 또는 replay의 마지막
transition에서는 실제 다음 화면을 확정할 수 없어 그림에 unavailable을 표시하고
NPZ의 `next_obs_valid=False`로 기록합니다. 해당 NPZ의 0 배열을 실제 화면으로 해석하면 안 됩니다.
경계 판별은 체크포인트에 저장된 termination을 사용합니다.

## 각 neighbor의 이후 화면 확인 후 선택해서 비교하기

새 분석에서는 기본적으로 **모든 neighbor의 t+1, t+2, …, t+6**을 저장합니다.
`events/anchor/point_event.png`, `events/neighbor_01/point_event.png`부터 확인하거나
`all_point_events.png`에서 전체를 비교하세요. 기본 target 16이면 anchor 1개와 최대 neighbor 15개입니다.

각 그림의 **t=0은 실제 검색된 관측의 위치**입니다. neighbor마다 다시 득점 시점을 찾아 정렬하지 않습니다.
neighbor가 득점 장면이 아닐 수도 있으며, 양의 reward가 실제로 있는 경우에만 SCORING ACTION / AFTER POINT를 표시합니다.
각 화면 아래 action·reward는 그 화면에서 출발하는 transition에 해당합니다.
t+6은 replay/environment step 6개 뒤이며, 원본 Atari 비디오 프레임 6개 뒤라는 의미는 아닙니다.
중간에 저장된 termination이나 replay 끝을 만나면 그 이후 칸은 unavailable로 표시합니다.

이미 이전 버전으로 분석 결과를 만들었다면 다음 명령으로 모든 neighbor의 연속 그림을 추가할 수 있습니다.
`analysis.json`의 기존 anchor·neighbor 인덱스와 checkpoint replay만 사용하므로 PCA·점수·검색은 다시 수행하지 않습니다.

```bash
python probing/pong_retrieval_events.py \
  --analysis results/pong_retrieval_analysis \
  --output results/pong_all_events \
  --future-steps 6
```

그림을 확인한 뒤 예를 들어 neighbor 2, 5, 11을 선택했다면:

```bash
python probing/pong_retrieval_events.py \
  --analysis results/pong_all_events \
  --neighbors 2 5 11 \
  --past-steps 2 --future-steps 6 \
  --layout rows \
  --output results/pong_selected_events
```

`comparison.png`와 `comparison.pdf`는 `all_point_events.png`와 같은 배치입니다.
각 행에서 시간은 왼쪽에서 오른쪽으로 진행하고, **anchor / neighbor 02 / neighbor 05 / neighbor 11**을
위에서 아래로 배치합니다. 기본값이 `--layout rows`이므로 이 옵션은 생략해도 됩니다.
이전에 사용한 명령에 `--layout columns`가 있다면 삭제하거나 `--layout rows`로 바꾸세요.
anchor는 항상 포함되며,
선택 번호와 인덱스는 `selection.json`에 남습니다. 이 명령은 GPU나 모델 추론을 사용하지 않습니다.
새 분석 결과에는 연속 프레임이 저장되므로 그 범위 내에서 재배치할 때 checkpoint도 읽지 않습니다.
새 분석 결과 디렉터리를 바로 `--analysis`에 지정해도 됩니다.

`--past-steps 0 --future-steps 2`로 현재 화면과 직후 2스텝만 비교할 수도 있습니다.
처음 저장한 범위보다 더 긴 시간이 필요하면 원본 분석 디렉터리(`analysis.json`이 있는 곳)를
`--analysis`로 지정하세요. 해당 checkpoint의 replay에서 같은 인덱스의 더 긴 구간을 읽습니다.
원본 checkpoint가 이동했다면 `--checkpoint 새경로`를 추가하세요.

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
- `--seed 1234`: 분석 난수를 고정합니다. 생략하면 실행마다 새 seed를 사용합니다.
- `--past-steps 2 --future-steps 6`: 각 anchor/neighbor 연속 그림의 과거·미래 범위. 점수 후보 범위인 `--post-score-steps`와 별개입니다.
- `--rebuild-chunk-size 1024`: global rebuild의 encoder 처리 단위. 기본값은 1024입니다.
- `--multiplier 16`: hash 재확인 전 후보 샘플 수의 배수. 생략하면 저장된 설정을 사용합니다.
- `--min-score 5.0`: trigger를 통과한 후보에 추가로 적용할 최소 점수입니다.
  저장된 trigger threshold보다 낮게 지정해도 trigger 조건을 완화하지 않습니다.
- `--latent-mode probs`: categorical sample 대신 확률 벡터를 critic 입력에 사용하는 별도 분석입니다.
  이는 여러 latent sample에서 평균 낸 기대 value가 아닙니다. hash encoding 모드는 저장된 설정을 유지합니다.
- `--action-labels NOOP FIRE RIGHT LEFT RIGHTFIRE LEFTFIRE`: Pong minimal action ID 0~5에 이름 추가.
  기본 그림에는 action ID를 표시합니다.

trigger 조건을 만족하며 neighbor를 반환하는 후보가 없으면 `candidate_scores.csv`와 `analysis.json`을
남기고 종료합니다. 새 무작위 실행, 더 넓은 `--post-score-steps` 또는 더 큰 `--multiplier`를 사용할 수 있습니다.
조건 미달 후보를 대신 선택하거나 저장된 trigger threshold를 낮추지 않습니다.

요청에 따라 이 스크립트는 작성만 했으며 실행·검증하지 않았습니다.
