# 성능 최적화 옵션

> 이 문서의 구현·측정 기록은 2026-09-29의 기존 두 옵션에 관한 것이다.
> 현재 YAML은 두 옵션과 추가 실행 최적화를 활성화한다.
> 2026-09-30 설정과 검증 결과는 [execution_optimizations.md](execution_optimizations.md)를 참고한다.

`config_files/STORM.yaml`에 두 옵션을 추가했다. 기본값과 옵션이 없는 기존 설정 파일의 동작은 모두 기존 버전이다.

```yaml
Performance:
  VectorizedReplaySampling: False
  DisableDistributionValidation: False
```

- `VectorizedReplaySampling: True`: GPU 리플레이 샘플링을 일괄 인덱싱으로 실행한다. 기존 NumPy 난수 추출 순서, 배치 순서, dtype, 정규화, 반환 레이아웃을 유지한다. 여러 환경과 demonstration 혼합 배치를 지원한다. CPU에 버퍼를 저장하는 설정은 기존 경로를 사용한다.
- `DisableDistributionValidation: True`: 해당 실행 프로세스에서 PyTorch 확률분포의 기본 유효성 검사를 끈다. `OneHotCategorical` 내부의 `Categorical`에도 적용된다. 계산식·샘플링 함수는 그대로이고, 잘못된 입력의 오류 검출 동작은 달라질 수 있다. 명시적으로 `validate_args=True`를 준 분포의 검사는 유지된다.
- 각 옵션은 독립적이다. 둘 다 `True`이면 두 최적화를 모두 사용한다.

## 기존 코드 보존

`False`일 때 GPU·CPU 샘플링은 기존 함수 본문을 그대로 실행한다. `sample`과 `sample_external`은 새 경로로 가는 조건문만 앞에 추가했고, 그 뒤 기존 본문은 수정 전 Git HEAD와 원문 및 AST가 동일함을 확인했다.

검사 해제 옵션이 `False`이면 PyTorch 기본값을 설정하는 함수도 호출하지 않는다. 원래 기본값과 `python -O`의 기존 동작을 보존한다. `agents.py`와 `sub_models/world_models.py`는 파일 전체가 수정 전과 바이트 단위로 동일하다.

옵션은 `train.py`와 `eval.py`가 시작할 때 읽는다. 실행 중인 프로세스의 설정을 파일 변경만으로 바꾸지는 않는다. 다른 설정을 적용하려면 해당 설정으로 새 프로세스를 실행한다. 공통 준비 학습의 저장 설정과 후속 분기에도 옵션이 전달된다.

## 사용 예

두 최적화를 모두 사용하려면 YAML에서 두 값을 `True`로 바꾸거나, 기존 학습 명령 뒤에 다음 인자를 붙인다.

```bash
Performance.VectorizedReplaySampling True Performance.DisableDistributionValidation True
```

기존 경로를 사용하려면 둘 다 `False`로 설정한다. 설정에 `Performance` 항목이 없더라도 두 값은 `False`로 로드된다.

## 검증 결과 — 2026-09-29

유휴 RTX A6000에서 회귀 테스트 9개가 모두 통과했다.

- 이전 설정 파일 로딩, 기본값, 실행 인자 덮어쓰기, 분기용 설정 저장·재로딩.
- 검사 해제 옵션이 중첩 분포에 적용되고, 기존 옵션에서는 전역 설정 호출이 없는지 확인.
- 여러 환경, demonstration 혼합·전용 배치, CPU 저장 경로, 실제 16×64 및 1024×8 배치에서 샘플 데이터·레이아웃·난수 상태 비교.
- 기존 실행 반복과 세 가지 최적화 조합을 동일한 초기 상태에서 비교. 실제 월드 모델·정책 모델로 연속 2회 업데이트하면서 출력, 행동, 손실, 기울기, 파라미터, 옵티마이저, GradScaler, 반환값 EMA, 난수 상태가 모두 **비트 단위로 일치**했다.

위 결과는 테스트한 환경·입력·업데이트 구간에 대한 검증이다. 전체 10만 step 학습이나 다른 하드웨어·PyTorch 버전의 비트 단위 동일성을 보장하는 결과는 아니다.

테스트 실행 명령:

```bash
CUDA_VISIBLE_DEVICES=<비어 있는 GPU UUID> python -m unittest discover -s tests -v
```

구현된 경로를 사용한 분리 성능 측정:

| 구간 | 기존 경로 중앙값 | 최적화 경로 중앙값 |
|---|---:|---:|
| 샘플링 16×64 | 0.528 / 0.543ms | 0.435 / 0.434ms |
| 샘플링 1024×8 | 12.237 / 13.300ms | 2.638 / 2.632ms |
| 미래 궤적 생성 1024×16 | 95.751 / 95.656ms | 69.672 / 69.762ms |

각 값은 두 차례 측정 결과다. 준비 실행 4회 후 12회의 중앙값을 사용했다. 미래 궤적 생성은 분포 검사 유무만 비교했다. 합성 데이터와 미학습 모델의 부분 측정이므로 전체 학습 시간 개선율과 같지 않다.

측정 산출물: `/tmp/storm-hero-performance-implemented-20260929/`.
재현 스크립트: `diagnostics/benchmark_shared_paths.py`.
