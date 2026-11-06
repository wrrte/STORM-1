# 논문 main 시드의 Random Neighbor ablation

2026-11-07: 논문 main performance의 26게임 / 103개 공통 시드를 job_queue_3090.txt에 등록했다.

- 같은 게임·시드의 재사용 가능한 retrieval 이전 warmup이 있는 30건은 resume.
- 나머지 73건은 처음부터 학습. 기존 완료 결과가 있는 시드도 이번 실험명으로 새 작업을 등록했다.
- 설정: neighbor_source=random, warmup_steps=50000, anchor_offset=-2, target=16, anchor_weight=0.0625, anchor_weight_r=null.
- Freeway는 Freeway.pkl demonstration을 사용한다.
- 학습은 102000 step까지 진행하고 100000-step checkpoint를 20 episode / BF16·FP32로 최종 평가한다.
- 이번 작업은 고정 YAML을 사용하므로 기본 STORM.yaml의 이후 수정에 영향을 받지 않는다.

3090 GPU가 있는 장비에서 해당 GPU 번호의 worker를 실행한다 (GPU 0 예시):

```bash
cd /home/choemj/STORM-1
bash 0_train.sh
```

queue를 명시하려면 `bash 0_train.sh job_queue_3090.txt`를 사용한다.
명령에 STORM Python의 절대 경로를 넣어 worker 실행 전 conda 활성화가 필요하지 않다.

각 작업의 평가 결과는 jobs.csv / plan.json에 기록된 `final_evaluation_path`에 저장된다.
예: `runs/Alien-paper-random-w50000-m2-seed1-20261107_randomneighor/final_evaluation.json`.
TensorBoard 및 W&B에는 BF16과 FP32 평가가 각각 기록되며, final_evaluation.json은 기존 기본 지표인 BF16 결과를 담는다.

`jobs.csv`는 전체 작업표, `plan.json`은 세부 검증 내역, `commands.txt`는 등록한 명령의 사본이다.
`queue_before.txt`는 등록 전 queue의 사본이다. source queue는 worker가 실행하면서 소비하므로 commands.txt로 등록 내용을 확인할 수 있다.

재사용한 warmup은 같은 게임·시드 및 호환 설정을 검증했지만 논문 원본의 정확한 trajectory를 보장하지는 않는다.
또한 과거 숨김 시트는 offset -2/-3을 묶어 집계했으며 이번 실험은 사용자가 요청한 -2로 통일했다.
논문의 시드별 retrieval 점수와 원본 run ID/offset도 plan.json에 보관했다.

명령 인자 및 child config 103건, warmup 상태와 replay archive 37개, Atari ROM 26개를 점검했다.
실제 GPU 학습은 시작하지 않았다.
