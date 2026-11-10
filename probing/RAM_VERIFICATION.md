# 네 게임의 RAM 추출 및 프레임 검증

`ram_states.py`는 Seaquest, Frostbite, Pong, Boxing의 RAM 한 행에서 의미 있는
수치와 객체를 추출한다. `verify_ram_states.py`는 프레임과 추출 결과를 나란히
표시하는 HTML 리포트, 이미지 모음, JSON, CSV, NPZ를 만든다.
모델 checkpoint, GPU, OCAtari 설치는 필요 없다. `conda activate storm`에서 실행한다.

## 바로 결과 보기

이번 작업에서 게임별 2,400 decision steps를 직접 수집했고 16장씩 표시했다.
저장된 Seaquest 1,473행, Frostbite 1,854행에도 별도 리포트를 생성했다.
아래 경로는 STORM-1 기준이다. `report.html`을 브라우저로 열거나
`contact_sheet.jpg`를 이미지 뷰어로 열면 된다. 서버는 필요 없다.

| 데이터 | 리포트 | 프레임 모음 |
| --- | --- | --- |
| 새 Seaquest | `results/ram_verification/seaquest/report.html` | 같은 폴더의 `contact_sheet.jpg` |
| 새 Frostbite | `results/ram_verification/frostbite/report.html` | 같은 폴더의 `contact_sheet.jpg` |
| 새 Pong | `results/ram_verification/pong/report.html` | 같은 폴더의 `contact_sheet.jpg` |
| 새 Boxing | `results/ram_verification/boxing/report.html` | 같은 폴더의 `contact_sheet.jpg` |
| 기존 Seaquest | `results/ram_verification/existing/seaquest/report.html` | 같은 폴더의 `contact_sheet.jpg` |
| 기존 Frostbite | `results/ram_verification/existing/frostbite/report.html` | 같은 폴더의 `contact_sheet.jpg` |

왼쪽이 원본, 오른쪽이 RAM 박스다. 각 번호는 아래 객체 표와 연결된다.
표에는 ID, 종류, `xywh`, 근거 RAM 주소/값, lane/빙판 색 등의 속성이 나온다.
수치가 바뀌는 전후 프레임, 시간상 균등한 프레임, seed로 고정한 임의 프레임을 섞는다.
`--frames`를 지정하면 원하는 행을 그대로 확인할 수 있다.

## 실행

```bash
cd /home/choemj/STORM-1
conda activate storm

# 새 데이터 수집 + 네 게임 검증. 다시 수집할 때는 새 output 경로를 사용한다.
python probing/verify_ram_states.py --game all --steps 2400 --samples 16 \
  --output results/ram_verification/new_run

# 기존 데이터 사용: 에뮬레이터 실행 없이 모든 RAM 행을 추출하고 16장 표시
python probing/verify_ram_states.py --game seaquest \
  --data probing_data/Seaquest_20260822_133245.npz \
  --output results/ram_verification/existing

python probing/verify_ram_states.py --game frostbite \
  --data probing_data/Frostbite_20260822_201417.npz \
  --output results/ram_verification/existing

# 이미 수집한 데이터에서 특정 프레임 재검증
python probing/verify_ram_states.py --game pong \
  --data results/ram_verification/pong/rollout.npz \
  --frames 14 15 994 995 --output results/ram_verification/pong_selected

# 단위 테스트
python -m unittest probing.test_ram_states -v
```

`--steps`는 의사결정 횟수다. 기본 `--frame-skip 4`에서 한 decision은 최대 4개의
ALE 프레임을 진행한다. `frame_index`와 `--frames`는 NPZ **행 번호**이며 ALE frame
number와 다르다. 새 수집 파일에는 둘 다 저장한다. 랜덤 정책으로 수집하므로
늦은 레벨이나 드문 상황을 충분히 보장하지는 않는다. Frostbite 이글루 0..16단계는
진행이 더 다양한 기존 데이터도 함께 확인하는 것이 좋다.

## 무엇을 추출하는가

주소는 모두 0부터 시작한다. 좌표는 원본 ALE 160×210의 왼쪽 위 기준 픽셀 값이며
화면의 객체 중심 좌표가 아니다. 원본 전체가 resize된 관측은 같은 비율로 표시한다.
crop/letterbox/회전한 관측에는 별도 좌표 변환이 필요하다.

| 게임 | 주요 RAM 주소 | 추출 내용 |
| --- | --- | --- |
| Seaquest | 70, 97 | 플레이어 `(x, y+33)` |
| Seaquest | 71..74, 89..92 | 잠수부/적 미사일 판별 및 위치. lane 0은 아래, 3은 위. 잠수부 `y=141-24*lane` |
| Seaquest | 30..33, 36..39, 89..93 | 적 formation, 상어/잠수함 종류 및 위치 |
| Seaquest | 62, 102, 59 | 탑승 잠수부 수, 산소, 예비 목숨. 수중 잠수부와 탑승 인원은 서로 다른 값 |
| Frostbite | 102, 100, 106..107 | 플레이어 `(x, y+29)` 및 사망/가라앉기 상태에 따른 박스 |
| Frostbite | 77 | 이글루 단계: 255→0, 0..15→1..16. 그 외는 null+warning |
| Frostbite | 30..34, 43..46 | 빙판 배치/위치와 흰색 여부, 오른쪽 경계에서 왼쪽으로 이어지는 박스 |
| Frostbite | 35..38, 84..91, 104 | 새/게/조개/물고기, 곰의 종류/위치 |
| Frostbite | 76, 101 | 예비 목숨, 온도 raw 및 BCD 해독 값 |
| Pong | 49, 54 | 공 `(x-49, y-14)` 및 공 유무 |
| Pong | 51, 50 | 오른쪽 플레이어/왼쪽 상대 패들. 위/아래 끝에서 줄어드는 박스 처리 |
| Pong | 14, 13 | 플레이어/상대 점수 |
| Boxing | 32..35 | 흰 선수 `(r32+5,r34+38)`, 검은 선수 `(r33+4,r35+38)` |
| Boxing | 18..19 | 점수 raw 및 BCD 해독 값. 예: raw 36 = 0x24 → 화면의 24점 |
| Boxing | 55, 57, 61, 59 | 플레이어 왼팔/오른팔, 상대 왼팔/오른팔의 raw 코드 |

Boxing의 박스는 몸통 영역으로 뻗은 주먹 전체를 포함하지 않는다. 팔 코드의 의미를
거리나 각도라고 가정하지 않는다. 유효한 BCD가 아닌 점수는 특별 상태일 수 있으므로
null로 남긴다. 온도도 원본 바이트를 함께 보존한다.
Seaquest의 플레이어 미사일, 산소 바, 탑승 잠수부 아이콘도 표시한다.
모든 게임의 모든 HUD/애니메이션/내부 변수를 복원하는 추출기는 아니다.

## 시간 정렬과 실제 확인 방법

새 데이터에서는 `screen[t]`, `ram[t]`를 같은 에뮬레이터 상태에서 action 전에 읽는다.
`action[t]`, `reward[t]`, `termination[t]`는 그 뒤 transition에 해당한다.
`obs[t]`는 마지막 두 프레임의 max-pooling 후 64×64로 축소한 STORM 형식이다.
환경 설정은 full action space, frame skip 4, sticky action probability 0이다.
리포트의 기본 화면은 `screen`이며, STORM `obs`도 펼쳐 볼 수 있다.

1. 플레이어·공·잠수부 박스가 실제 객체와 같은 위치에 있는지 본다.
2. 잠수부 획득/점수 변경/이글루 변화 전후 프레임에서 scalar가 함께 변하는지 본다.
3. Seaquest에서는 같은 RAM 슬롯이 적 미사일로 사용될 때 잠수부로 잘못 세지 않는지 본다.
4. 죽는 중, 공이 없는 중, 패들이 위아래 끝에 있는 경우, 빙판이 화면 경계를 넘는 경우도 본다.

Atari sprite flicker 때문에 특정 마지막 프레임에 RAM상의 물체가 안 그려질 수 있고,
max-pooling 관측에는 이전 위치의 잔상도 남을 수 있다. 한 장의 픽셀 유무만으로
RAM 오류라고 판단하지 말고 앞뒤 프레임과 `obs`도 비교한다.
기존 파일은 수집 코드/시간 정렬을 확인하지 못했으므로 리포트에도 정렬 미검증으로
표시한다. 특히 기존 파일의 첫 화면은 검은 초기 화면이므로 의미 검증에서 제외한다.
warning 0개는 범위/알려진 코드 검사 결과이며 의미적으로 완전히 맞다는 보증은 아니다.

### 현재 환경에서 확인한 getRAM 문제

이 작업 환경은 NumPy 2.2.6 + ale-py 0.8.1이다. 인자 없는 `ale.getRAM()`이 128개
바이트를 모두 첫 바이트와 같은 값으로 반환하는 현상을 직접 재현했다. 반면
`buffer=np.empty(128,dtype=np.uint8); ale.getRAM(buffer)`는 정상 바이트를 읽는다.
`read_ale_ram()`은 버퍼를 넘기는 API를 우선 사용하고, 이 API가 없는 새 버전에서만
인자 없는 API로 전환한다. 상수 RAM 행은 즉시 거부한다. 기존 probing 데이터는
서로 다른 바이트로 저장되어 있다. 환경 패키지나 다른 수집 코드는 변경하지 않았다.

NumPy 2 지원 변경은 [ALE v0.9.1 release notes](https://ale.farama.org/release_notes/#ale-v0-9-1)에도
기록되어 있다. 이 환경의 문제는 직접 확인한 결과이며, 모든 구버전 환경에서 같은
현상이 난다고 가정하지 않는다.

## 저장 파일과 이후 retrieval 분석

각 게임 폴더에 다음이 생긴다.

- `report.html`, `contact_sheet.jpg`: 사람이 확인할 리포트와 프레임 모음.
- `frames/frame_XXXXXX.png`: 원본/오버레이 한 쌍. `_raw`, `_overlay`, 선택시 `_obs`도 저장.
- `frames/frame_XXXXXX.json`: 선택한 행의 전체 128바이트 RAM과 추출 상태.
- `states.jsonl`: **모든 행**의 scalar, 객체 종류/위치/근거 RAM, warnings.
- `features.csv`: 모든 행의 scalar/객체 종류 one-hot/위치/크기/존재 여부.
- `features.npz`: 같은 수치를 `values`, 열 이름을 `names`, 유효 mask를 `valid`로 저장.
- `metadata.json`: 입력 파일, 좌표계, 표시한 행 번호, warning 행, 매핑 출처, 수집 설정.
- `rollout.npz`: 새 수집 시에만 생성. 기존 파일에서 분석할 때는 원본을 그대로 읽음.

객체가 없는 위치는 NaN/CSV 빈칸이며 존재 여부는 0이다. NaN을 좌표 (0,0)으로
해석하면 안 된다. 열은 해당 데이터에 나타난 객체들의 합집합으로 만들어지므로
여러 파일 간 비교 시 `names`로 정렬한다. 객체 ID는 RAM 슬롯이며 영구 추적 ID가
아니다. 슬롯의 순서가 다른 동등한 객체 집합은 종류/좌표 기반 매칭이 필요하다.
종류 one-hot과 presence는 누락 시 0, 나머지 누락 수치는 NaN이다.

직접 사용할 API:

```python
import numpy as np
from probing.ram_states import extract_state, read_ale_ram

with np.load("results/ram_verification/pong/rollout.npz") as data:
    state = extract_state("pong", data["ram"][15])
    print(state["scalars"])
    print(state["objects"])

# 학습/평가 환경에 수집을 붙일 때:
# obs와 같은 상태에서, env.step(action) 전에 호출하고 row/episode ID도 함께 보존
# ram = read_ale_ram(env.unwrapped.ale)
```

한 시점 RAM의 위치만으로 Pong 공의 속도/진행 방향까지 복원하지 않는다. 이후
retrieval의 상태 유사도를 평가할 때는 같은 episode의 앞뒤 RAM을 이용한 속도,
게임별 정규화와 가중치, 객체 집합 매칭 등 평가 기준을 따로 정해야 한다.
현재 exports를 그대로 Euclidean distance에 넣은 값을 완성된 semantic metric이라고
해석하지 않는다.

추출 검증을 위해 지금 추가로 제공할 파일은 없다. **실제 retrieval neighbor의 유사도**를
이어 분석하려면 query와 neighbor가 어느 run/episode/frame/replay row인지 연결할 수 있는
정보와 그 시점 RAM이 필요하다. 일반 STORM replay는 RAM을 저장하지 않으므로,
checkpoint나 이미지/latent만으로 과거 RAM을 정확히 복구할 수 있다고 가정하지 않는다.

## 매핑 출처

기존 `old/verify_diver_ram.py`, `probing/train_probing_mlp.py`의 Seaquest 규칙을 확인했고,
네 게임의 [OCAtari RAM readers](https://github.com/k4ntz/OC_Atari/tree/master/ocatari/ram)를
2026-11-10에 대조했다:
[Seaquest](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/seaquest.py),
[Frostbite](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/frostbite.py),
[Pong](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/pong.py),
[Boxing](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/boxing.py).
매 행을 독립적으로 해독해, 시간순이 아닌 retrieval neighbor에도 적용할 수 있다.
산술 전에 정수형으로 변환해 uint8의 덧셈/뺄셈 overflow도 방지한다.
