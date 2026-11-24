이 문서는 후속 코드 수정 전의 분석 기록이다. 이후 사용자 요청으로 `STORM/train.py`의 Both 분기를 변경했다. 현재는 warmup 기준에 도달한 뒤 실제 episode 종료까지 공통 학습을 계속하고, 종료 transition의 update를 마친 뒤 분기한다. 강제 terminal 덮어쓰기는 제거했고 hash rebuild는 실제 분기 직전 한 번 수행한다. 현재 방식은 NumEnvs=1을 지원하며, 아래의 과거 코드 위치와 32개 테스트 결과는 수정 전 상태를 가리킨다. 수정 후 관련 테스트는 37개가 통과했다.

`Both`의 분기에는 단독 학습과의 동등성을 깨는 문제가 있다. 그러나 이번에 확인한 Gopher 3개 시드의 낮은 점수를 “체크포인트가 잘못 복원되어 생긴 성능 붕괴”로 결론 내릴 증거는 없다. 모델·optimizer 복원 검사는 통과했고, 실제 GPU에서 실행한 세 쌍의 첫 world-model loss도 각각 완전히 일치했다. 우선 수정·분리 검증해야 할 것은 **환경/context 보존, 분기 경계의 sequence 처리, 난수 스트림 분리**다.

분석일: 2026-09-20. 학습 코드는 수정하지 않았다. 이 폴더에는 진단 스크립트, W&B에서 읽은 데이터, 비교 표와 그림만 추가했다.

**실제 실행에서 확인한 사실**

| Seed | GPU | 공통 warmup 마지막 10개 학습 episode 평균 | Both → False 평가 | Both → True 평가 |
|---|---|---:|---:|---:|
| 6000 | RTX A6000 | 1,554 | 2,486 | 1,024 |
| 6010 | RTX A6000 | 1,044 | 1,250 | 927 |
| 9999 | TITAN RTX | 1,170 | 2,594 | 1,005 |

학습 episode 평균과 최종 평가는 서로 다른 측정이므로 표의 열들을 직접적인 성능 증감량으로 해석하면 안 된다. 최종 평가는 각각 20개 episode이고, 저장된 개별 return의 산술평균이 W&B의 `eval/episode_avg_return`과 일치한다.

- 6000: [warmup s12t06bx](https://wandb.ai/choemj-kaist/STORM/runs/s12t06bx), [False w30fmuzh](https://wandb.ai/choemj-kaist/STORM/runs/w30fmuzh), [True ori7lpmm](https://wandb.ai/choemj-kaist/STORM/runs/ori7lpmm).
- 6010: [warmup p9dfsvag](https://wandb.ai/choemj-kaist/STORM/runs/p9dfsvag), [False 3d8lc1uk](https://wandb.ai/choemj-kaist/STORM/runs/3d8lc1uk), [True tepmvj6g](https://wandb.ai/choemj-kaist/STORM/runs/tepmvj6g).
- 9999: [warmup 9uwipoit](https://wandb.ai/choemj-kaist/STORM/runs/9uwipoit), [False hrca6yyg](https://wandb.ai/choemj-kaist/STORM/runs/hrca6yyg), [True uz4318p4](https://wandb.ai/choemj-kaist/STORM/runs/uz4318p4).

실제 자식 프로세스의 CLI에는 `--branch_mode true/false`와 `--resume_from .../shared_warmup_50000`이 들어 있다. 부모와 두 자식의 설정은 `Retrieval.enable`을 제외하면 동일하다. 모두 `SampleMaxSteps=102000`, `NumEnvs=1`, `BufferWarmUp=1024`, `warmup_steps=50000`, `SaveEverySteps=50000`, `EvalMode=final_only`다. 따라서 자식 W&B run에 약 52,000개의 step만 있다는 사실은 학습 예산 누락을 뜻하지 않는다. 공통 warmup의 모델 업데이트 48,976회와 자식의 52,000회를 합하면 단독 실행과 같은 100,976회다.

세 쌍의 재개 후 `_step=0`에서 `WorldModel/total_loss`는 다음과 같다.

| Seed | False | True | 일치 |
|---|---:|---:|---|
| 6000 | 1.5272964239120483 | 1.5272964239120483 | exact |
| 6010 | 1.5578410625457764 | 1.5578410625457764 | exact |
| 9999 | 1.5452371835708618 | 1.5452371835708618 | exact |

이는 **두 자식이 같은 상태에서 시작한다는 증거**다. 환경을 유지한 단독 학습과 동일하다는 증거는 아니다. 두 자식 모두 같은 환경 초기화를 거치기 때문이다.

학습 곡선은 [gopher_training_curves.png](gopher_training_curves.png), 수치 요약은 [gopher_summary.csv](gopher_summary.csv)에 있다. warmup 후반부터 episode return은 낮은 수준이고, True 분기는 대체로 그 수준에 머물렀다. False의 마지막 10개 학습 episode 평균은 6000/6010/9999 순서로 1,792/1,400/2,124이고, True는 1,084/986/1,114다. 따라서 현재 로그는 “재개 시점의 일괄적인 급락”보다 “낮은 성능에서 시작한 이후 True가 충분히 개선되지 않는 현상”에 가깝다. 같은 시드·GPU·코드·설정의 단독 실행과 비교하지 않았으므로, warmup 수준 자체가 비정상이라고 단정할 수는 없다.

**1. 환경과 수집 context가 저장되지 않는다 — 확정된 동등성 결함**

`train.py:519–545`의 체크포인트에는 가중치, optimizer, scaler, EMA, replay, 전역 RNG와 retrieval 상태가 저장된다. 그러나 Atari/ALE 상태, 환경 자체의 RNG, action-space RNG, wrapper 상태, `current_obs/current_info`, `context_obs/context_action`, 진행 중인 episode의 `sum_reward`는 없다.

재개 시 `train.py:190–197`에서 환경을 새로 만들고 reset하며 context를 비운다. `train.py:296–299`는 빈 context에서 학습된 정책 대신 `action_space.sample()`을 사용한다. 원래 학습이 진행 중이던 게임 상태를 이어가지 않고, 처음 화면에서 random action으로 다시 수집한다. Python/NumPy/Torch/CUDA RNG를 복원하는 것만으로는 Atari 내부 상태와 action-space generator를 복원할 수 없다.

실제 ALE/Frostbite CPU 환경으로 100번 action을 수행한 경우, 진행 중인 환경은 frame 400이었지만 새 환경은 frame 0이었다. 새 관측은 최초 관측과 같고 진행 중 관측과는 달랐다. 이는 환경 reset 동작 검증이며 Gopher 점수에 대한 인과 실험은 아니다.

우선순위가 높은 수정은 ALE 상태를 `clone_state(include_rng=True)` 등으로 보존하고, wrapper 및 수집 루프 상태까지 함께 복원하는 것이다. Atari만 복원하고 context를 비우거나 reset된 `current_obs`를 쓰는 것도 불완전하다. NumEnvs > 1에서는 각 worker의 상태를 보존해야 한다. 현재 분기의 양쪽이 같은 reset을 겪는다는 사실만으로 단독 실행과의 동등성이 성립하지 않는다.

**2. 강제로 terminal을 넣어도 시퀀스 학습은 분기 경계를 넘는다 — 재현한 학습 데이터 문제**

`train.py:267–271`은 분기 직전 replay의 마지막 transition을 무조건 terminal=1로 덮어쓴다. 실제 게임이 끝나지 않았더라도 이 label이 바뀐다.

그런데 `replay_buffer.py:58–118`의 일반 sampler는 terminal을 기준으로 시퀀스를 나누지 않는다. `sub_models/world_models.py:452–465`의 Transformer mask는 causal mask뿐이고, dynamics/representation KL도 인접한 모든 프레임 쌍에 적용된다. 따라서 warmup의 게임 중간 화면 다음에 reset된 첫 화면이 붙는 시퀀스를 정상적인 시간 흐름으로 학습한다. terminal label은 일부 value bootstrap을 제한하지만 world-model의 KL target이나 attention context를 끊지 않는다.

실제 sampler로 재현한 결과는 관측 `[12, 13, 0, 1]`, terminal `[0, 1, 0, 0]`이었다. 여기서 `13 → 0`은 reset 경계인데도 시퀀스에 포함됐다. retrieval의 `_valid_context()`는 이런 경계를 검사하지만 일반 world-model/무작위 imagination sampler는 검사하지 않는다.

다만 기본 설정에서는 추가 경계가 한 곳이다. 충분히 데이터가 쌓인 뒤 length=64 시퀀스 중 이 경계를 가로지르는 시작점은 최대 63개이므로, 50k–100k buffer에서 직접 영향받는 시작점 비율은 약 0.06–0.13%다. 작은 교란이 RL에서 증폭될 수는 있지만, **이 결함만으로 여러 시드의 큰 성능 저하가 설명된다고 주장할 수는 없다**. 자연 episode 경계에도 관련 문제가 있으므로 Both 전용 결함과 공통 sampler 결함을 구분해야 한다.

환경을 정확히 복원하면 Both가 추가한 인공 경계를 없앨 수 있다. reset을 유지한다면 별도의 episode-boundary 표식과 boundary-aware sampling/masking이 필요하며, 인위적 수집 중단을 실제 MDP terminal과 구분해야 한다.

**3. Both의 False는 단독 False와 난수 경로가 다르다 — 재현한 재현성 문제**

`retrieval.py:79–80`에서 `Both`는 retrieval manager가 enabled인 상태다. warmup에는 검색을 사용하지 않지만 hash/statistics를 관리한다. `train.py:430–440`은 고정 warmup이 끝나기 직전, 기본 설정에서 step 49,999에 global rebuild를 실행한다. 단독 False는 이 rebuild를 하지 않는다.

`retrieval.py:560`의 `torch.pca_lowrank`는 랜덤 projection을 사용한다. 현재 설정은 50k frame < max_pca_samples 100k라서 `randperm`은 필요하지 않아도 PCA 자체가 RNG를 소비한다. CPU에서 실제 rebuild 호출 전후 Torch RNG가 달라지는 것을 확인했다. 따라서 Both → False는 retrieval을 사용하지 않더라도, 단독 False와 다른 training RNG 위치에서 다음 단계를 시작한다.

이 차이는 업데이트가 빠지거나 optimizer가 손상되는 오류가 아니라 동일 seed 비교를 방해하는 간섭이다. 시드 전체에서 성능을 특정 방향으로 낮춘다는 증거는 없다. training RNG와 retrieval/PCA용 RNG를 분리하거나 관련 RNG 상태를 보존·복구하는 방식이 필요하다.

**4. 과거 실험과의 설정 차이 중 SaveEverySteps를 무시하면 안 된다 — 공통 코드 문제를 추가 재현**

확인 가능한 단독 False 참조 run [Gopher_gu3gyrb0_1_X](https://wandb.ai/choemj-kaist/STORM/runs/gu3gyrb0)는 5,968점이지만, seed 1 / RTX PRO 6000 / 과거 코드 / SaveEverySteps=2500이다. 현재 세 쌍은 다른 seed, A6000 또는 TITAN RTX, SaveEverySteps=50000이다. 그러므로 이 점수 차이를 Both에만 귀속할 수 없다.

특히 `SaveEverySteps`는 단순 저장 간격이 아니다. `train.py:363–398`은 world-model update 후 `log_openloop_video()`를 부른다. `world_model.update()`는 `self.train()`을 호출하고, open-loop 함수는 eval mode로 전환하지 않는다. `world_model_imagine_data()`의 eval 전환은 그 이후다.

실제 모델에서 이를 재현했더니 open-loop 영상 1회 생성만으로 encoder의 각 BatchNorm `num_batches_tracked`가 +1, 테스트의 4-frame future 설정에서는 decoder가 +5 증가했다. Torch RNG도 변했다. 실제 기본 future length=16이면 decoder forward 횟수도 더 많다. Dropout 및 확률적 latent 샘플링도 실행된다. 따라서 2500 → 50000 변경은 BatchNorm 통계와 RNG 소비 주기를 바꾸므로 학습 경로를 바꾼다. 어느 주기가 더 좋은지는 이 진단으로 결정할 수 없다.

고정된 비교에서는 저장/영상 설정까지 맞추고, 영상 생성은 eval mode에서 수행하면서 학습 RNG를 보존해야 한다. 이 문제는 True/False/Both 모두에 있는 공통 문제이고, Both의 성능 하락 원인으로 확정된 것은 아니다.

현재 TITAN RTX 경로는 `sub_models/precision.py`에 의해 FP16, A6000 경로는 BF16을 사용한다. TITAN seed 9999에는 정밀도 차이가 있지만 A6000의 6000/6010도 낮으므로 FP16만을 유일한 원인으로 지목할 수 없다.

같은 seed의 과거 단독 True 점수는 로컬 보관 CSV에 6000=1,051(`gsvmbdpy`), 6010=2,234(`ic64hxjj`)로 남아 있다. 6000의 Both → True=1,024와는 큰 차이가 없다. 그러나 두 과거 run은 현재 W&B API에서 조회되지 않아 설정을 확인하지 못했다. 이 수치는 참고자료이며 엄밀한 대조군으로 사용하지 않았다. 다른 seed 9999의 일치하는 단독 대조군도 확보하지 못했다.

**문제로 확인되지 않은 부분과 검증 범위**

- optimizer의 Adam moment/step, world-model BatchNorm running buffers, slow critic, lower/upper EMA는 저장·복원된다. retrieval state를 불러와도 수신 branch의 enable flag는 유지된다. False가 잘못 True로 바뀌는 문제는 재현되지 않았다.
- 실제 STORM 모델 클래스의 작은 구성과 synthetic minibatch를 이용해 저장 후 원본/True 복원본/False 복원본에 동일한 다음 update를 적용했다. FP32 CPU에서는 weights, Adam state, EMA, RNG가 bitwise 동일했다. 실험 결과는 `diagnostics.txt`에 있다. GPU AMP scaler의 실제 동작까지 검사한 것은 아니다.
- 기존 `tests/test_retrieval_integration.py`의 32개 테스트는 모두 통과했다. 이 테스트들은 full Atari/GPU 학습 동등성을 검증하지 않는다. 기존 STORM 분기 테스트는 주로 분기 시점과 supervisor 호출을 확인한다.
- fixed warmup=50000에서 분기는 다음 collection/update 이전에 일어나므로 현재 코드의 학습 step에 50k 누락이나 한 step 중복은 확인되지 않았다.
- 기본 `final_only` 평가 경로는 단독/자식 실행에 동일하게 적용된다. standalone `eval.py:135–136`은 새 `world_model_final.pth`의 `final`을 int로 바꾸려다 실패하는 별도 호환성 버그가 있지만, 이번 W&B 내장 최종 평가 저하를 설명하지 않는다.
- W&B의 자식 `_step`은 0부터 다시 시작한다. 이를 environment step으로 직접 비교하면 곡선을 약 50k 잘못 정렬한다. `logger.tag_step` 복원은 TensorBoard tag별 counter를 위한 것이며 W&B global step을 복원하지 않는다.
- 부모·자식이 보고한 STORM git revision 사이의 train/model 관련 파일은 로컬 git diff에서 동일했다. 다만 실행 서버의 미커밋 변경이나 STORM 저장소 밖의 `retrieval.py`/`training_branches.py` 버전까지 원격 metadata만으로 보증할 수는 없다.

**해당 학습 서버에서 이어서 할 가장 가치 있는 검증**

해당 서버에서 실행하면 기존 `shared_warmup_50000/training_state.pt`, replay와 실제 GPU/라이브러리에 접근할 수 있으므로 다음 단계에 더 유리하다. W&B 분석은 현재 환경에서도 완료했다. 현재 분석 환경에는 CUDA device가 노출되지 않아 전체 GPU 학습 재현은 하지 않았다.

1. 원격의 `train.py`, 실제 import된 `retrieval.__file__`, `training_branches.__file__`, PyTorch/CUDA/AMP dtype와 실행 코드를 보존한다. STORM git commit만으로 외부 공통 모듈 버전을 추정하지 않는다.
2. 같은 GPU·seed·config에서 단독 True와 Both parent의 **분기 직전** weights/replay/RNG/EMA를 비교한다. 이때 기존 video/SaveEverySteps까지 동일해야 한다. warmup부터 다르면 reset 문제보다 앞선 코드·환경·난수 차이부터 추적한다.
3. 동일 minibatch와 RNG로 checkpoint 저장 전후의 첫 update를 CUDA에서 비교한다. 현재 CPU 검증의 GPU 확장이다. optimizer state tensor의 device, scaler scale/growth_tracker, BN buffers와 반환 EMA를 포함한다.
4. 환경/context 보존을 구현한 continuation과 현재 reset continuation을 동일 50k 상태에서 비교한다. 기존 checkpoint에는 ALE/context가 없으므로 기존 파일만으로 원래 ongoing environment를 정확히 복원할 수 없다. 정확한 비교를 위해서는 상태를 저장하도록 계측한 warmup을 한 번 새로 수행해야 한다.
5. 그 뒤 동일 seed·GPU·config의 단독 False/True와 Both 결과를 비교한다. 평가 환경 seed와 evaluation RNG도 맞춘다. 여러 변경을 한 번에 적용해 원인 추적을 어렵게 하지 않는다.

이 단계들은 분석 후의 검증 제안이다. 이 조사에서는 새 100k GPU 학습이나 production code 수정은 실행하지 않았다.

재현 명령은 프로젝트 상위 폴더에서 다음과 같다.

```bash
/home/ai2lab/miniconda3/envs/storm/bin/python -m unittest discover -s tests -p test_retrieval_integration.py -v
/home/ai2lab/miniconda3/envs/storm/bin/python STORM/results/both_audit_20260920/check_checkpoint_and_boundaries.py
/home/ai2lab/miniconda3/envs/storm/bin/python STORM/results/both_audit_20260920/summarize_gopher_runs.py
```

`fetch_gopher_runs.py`는 W&B 읽기 전용 조회 스크립트다. 설치된 W&B 0.28.1의 `scan_history(keys=...)`가 요청과 다른 column을 반환하거나 `_step` schema 오류를 내는 현상이 있어 최종 데이터는 GraphQL history 경로로 다시 수집했다. 잘못된 scan 결과는 비교에 사용하지 않았다. 학습 reward 곡선에는 별도 조회한 episode 기록을, entropy/loss 추세에는 run당 1500개 sampled row를 사용했다. 부모-자식 첫 update 일치는 0–20 구간의 별도 조회로 검사했다.
