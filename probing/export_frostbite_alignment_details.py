"""Export appendix-ready detail from saved Frostbite values; no model inference."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from frostbite_value_stats import correlations, make_blocks


METRICS = ('spearman', 'kendall')
META = ('frame_id', 'episode_id', 'episode_step', 'block_id')
MODEL_LABELS = {'random': 'Random neighbors (seed 11010)',
                'baseline': 'Baseline (seed 3710)', 'flash': 'FLASH (seed 3710)'}


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def save_csv(path, rows):
    rows = iter(rows)
    first = next(rows)
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def describe(values):
    values = np.asarray(values)
    x = values[np.isfinite(values)]
    result = dict(n=int(len(x)), undefined=int(len(values) - len(x)))
    if not len(x):
        return dict(result, mean=None, sd=None, minimum=None, q05=None, q1=None,
                    median=None, q3=None, q95=None, maximum=None, iqr=None,
                    positive=0, zero=0, negative=0, positive_fraction=None,
                    zero_fraction=None, negative_fraction=None,
                    positive_conditional_mean=None, negative_conditional_mean=None)
    minimum, q05, q1, median, q3, q95, maximum = np.quantile(x, [0, .05, .25, .5, .75, .95, 1])
    result.update(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else None,
                  minimum=float(minimum), q05=float(q05), q1=float(q1), median=float(median),
                  q3=float(q3), q95=float(q95), maximum=float(maximum), iqr=float(q3 - q1))
    for name, mask in [('positive', x > 0), ('zero', x == 0), ('negative', x < 0)]:
        result[name] = int(mask.sum())
        result[name + '_fraction'] = float(mask.mean())
        if name != 'zero':
            result[name + '_conditional_mean'] = float(x[mask].mean()) if mask.any() else None
    return result


def resampled_means(matrix, groups, selections):
    """Shared whole-cluster draws, with a separate valid denominator per column."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    ids, inverse = np.unique(groups, return_inverse=True)
    if len(ids) < 2 or selections.shape[1] != len(ids):
        raise ValueError('Need at least two clusters and one draw per observed cluster')
    valid = np.isfinite(matrix)
    sums = np.zeros((len(ids), matrix.shape[1]))
    counts = np.zeros_like(sums, dtype=np.int64)
    np.add.at(sums, inverse, np.where(valid, matrix, 0))
    np.add.at(counts, inverse, valid)
    denominators = counts[selections].sum(axis=1)
    if (denominators == 0).any():
        raise ValueError('Some bootstrap draws contain no defined observations')
    return sums[selections].sum(axis=1) / denominators


def match_reference(reference, metadata):
    """Match by frame ID, then verify episode/history/block identity."""
    ref_ids, frame_ids = reference['frame_id'], metadata['frame_id']
    if (len(np.unique(ref_ids)) != len(ref_ids)
            or len(np.unique(frame_ids)) != len(frame_ids)
            or not np.array_equal(np.sort(ref_ids), np.sort(frame_ids))):
        raise ValueError('Reference and Random must contain the same unique frame IDs')
    lookup = {int(frame): i for i, frame in enumerate(ref_ids)}
    order = np.array([lookup[int(frame)] for frame in frame_ids])
    for key in META:
        if not np.array_equal(reference[key][order], metadata[key]):
            raise ValueError(f'Mismatched context metadata: {key}')
    return {name: reference[name][order] for name in ('baseline', 'flash')}


def interval(draws):
    return np.quantile(draws, [.025, .975]).tolist()


def fmt(value, signed=False):
    return 'NA' if value is None else format(value, '+.4f' if signed else '.4f')


def cell(stat):
    return f"{fmt(stat['mean'])} [{fmt(stat['ci95'][0])}, {fmt(stat['ci95'][1])}]"


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def latex_table(label, caption, headers, rows, columns=None):
    columns = columns or 'l' + 'r' * (len(headers) - 1)
    return '\n'.join([r'\begin{table}[!htbp]', r'\centering', r'\small',
                      r'\caption{' + caption + '}', r'\label{tab:random11010_' + label + '}',
                      r'\begin{tabular}{' + columns + '}', r'\toprule',
                      ' & '.join(headers) + r' \\', r'\midrule']
                     + [' & '.join(map(str, row)) + r' \\' for row in rows]
                     + [r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def make_report(output, summary, episodes, profiles, comparisons, comparison_episodes):
    metrics = summary['models']['random']
    protocol = summary['protocol']
    counts = summary['context_accounting']
    pair_stats = {(c['reference'], c['metric']): c for c in comparisons}
    baseline_rho = pair_stats['baseline', 'spearman']
    lines = ['# Frostbite Random neighbors: seed 11010 상세 결과', '',
             '저장된 critic 예측만 분석했습니다. 모델 추론이나 추가 학습은 실행하지 않았습니다. '
             '원래 평가 파일을 보존하고 상세 결과를 이 디렉터리에 별도로 저장했습니다.', '',
             '## 체크포인트와 평가 문맥', '',
             f"- Run: `{protocol['run']}`",
             '- Variant: Random neighbors; salience는 value_diff × TD-error의 multiply 규칙을 유지하고 이웃만 무작위로 선택합니다.',
             '- 학습 seed: 11010; world model/critic 체크포인트: 100,000 agent steps.',
             '- 최종 정책 평가: BF16, 20 episodes 평균 2177.0점. FP32 기록은 2207.5점입니다. '
             '이 정책 평가 에피소드와 아래 critic 진단 에피소드는 서로 다른 데이터입니다.',
             '- 문맥 수집 정책: Baseline seed 3710의 100k 체크포인트. Random 정책으로 새로 수집한 문맥이 아닙니다.',
             '- 수집/평가 난수 seed: 2027; action repeat 4; 입력 64 × 64 RGB. '
             '20,000개는 agent-step 관측 수이며 raw emulator frame 수가 아닙니다.',
             '- 관측 에피소드: 20개(19개 완료, 마지막 1개 부분 관측).',
             '- 블록: 같은 원래 RAM igloo stage가 유지되는 연속 관측, 최대 64개, 에피소드 경계 통과 금지.',
             '- 원래 igloo stage 라벨은 RAM[77]=255이면 0, 그 외에는 RAM[77]+1입니다.',
             '- 이글루는 모든 블록 관측의 `[10:17, 42:58, :]` 영역을 7 × 16 RGB 템플릿 17개(0–16단계)로 교체합니다. '
             '나머지 픽셀과 행동 이력은 단계 간 동일하며 배경 노이즈는 없습니다.',
             '- 블록마다 앞 8개 관측으로 문맥을 초기화하고 이후 시점의 critic 값을 계산합니다. '
             '블록·단계별 난수 초기화는 평가 프로토콜에 기록된 방식을 따릅니다.',
             '- 각 문맥·단계는 latent draw 1회이며 반복 latent 평균이 아닙니다.', '',
             md_table(['문맥 구성 항목', '개수'], [
                 ['수집 관측', counts['collected_frames']], ['후보 블록', counts['candidate_blocks']],
                 ['길이 9 미만 제외 블록', counts['excluded_short_blocks']],
                 ['짧은 블록으로 제외된 관측', counts['excluded_short_frames']],
                 ['유효 블록', counts['valid_blocks']], ['문맥 초기화 관측', counts['warmup_frames']],
                 ['평가 문맥', counts['evaluated_contexts']], ['문맥 × 단계 critic 예측', counts['context_stage_predictions']]]), '',
             '## 정렬 점수 정의와 평균·95% CI', '',
             '문맥 i마다 stage 인덱스 (0,…,16)과 critic 값 (v_i(0),…,v_i(16)) 사이의 Spearman ρ와 Kendall τ_b를 구합니다. '
             '그 뒤 문맥별 상관계수를 같은 가중치로 평균합니다. ties는 SciPy의 기본 순위 처리를 사용합니다. '
             '17단계 예측이 상수이면 상관계수는 정의되지 않으므로 0으로 바꾸지 않고 제외합니다.', '',
             '20개 에피소드 ID를 복원 추출하는 bootstrap 10,000회(seed 2027)를 사용합니다. '
             '선택된 각 에피소드의 모든 문맥과 반복 선택 횟수를 유지하고, 매 회 선택된 문맥 수를 분모로 다시 계산합니다. '
             '2.5/97.5 percentile이 95% CI입니다. 아래 모든 bootstrap 결과의 에피소드 선택을 '
             '`bootstrap_replicates.npz`에 저장했습니다.', '',
             md_table(['지표', '평균 [95% CI]', '유효 문맥', '정의 불가'],
                      [[m, cell(metrics[m]), metrics[m]['n'], metrics[m]['undefined']] for m in METRICS]), '',
             '## 문맥별 분포', '',
             md_table(['지표', 'SD', '최솟값', '5%', 'Q1', '중앙값', 'Q3', '95%', '최댓값', 'IQR'],
                      [[m] + [fmt(metrics[m][k]) for k in ('sd', 'minimum', 'q05', 'q1', 'median', 'q3', 'q95', 'maximum', 'iqr')]
                       for m in METRICS]), '',
             'SD는 문맥별 상관계수의 표본 표준편차(ddof=1)입니다. 분포의 사분위수는 문맥 간 변동을, '
             'CI는 문맥 가중 평균의 에피소드 재표집 불확실성을 나타냅니다.', '',
             md_table(['지표', '양수 문맥', '0 문맥', '음수 문맥', '양수 조건부 평균', '음수 조건부 평균'],
                      [[m] + [f"{metrics[m][k]:,} ({100 * metrics[m][k + '_fraction']:.2f}%)" for k in ('positive', 'zero', 'negative')]
                       + [fmt(metrics[m]['positive_conditional_mean']), fmt(metrics[m]['negative_conditional_mean'])] for m in METRICS]), '',
             '## 에피소드별 결과', '',
             'Episode 열은 읽기 쉽게 1–20으로 표시하며 원자료 episode_id는 0–19입니다. '
             '문맥 ID와 블록 ID도 원자료의 0-based 인덱스를 그대로 CSV에 저장했습니다. '
             '아래 에피소드는 Baseline이 수집한 진단 문맥입니다.', '',
             md_table(['Episode', '수집 관측', '평가 문맥', '블록', '완료', '평균 ρ', '평균 τ_b'],
                      [[e['episode'], e['collected_frames'], e['evaluated_contexts'], e['blocks'], e['completed'],
                        fmt(e['spearman_mean']), fmt(e['kendall_mean'])] for e in episodes]), '',
             md_table(['지표', '양수 episode 평균 개수', 'episode 평균 최솟값', '최댓값', 'episode 평균의 단순 평균'],
                      [[m, f"{summary['episode_summary'][m]['positive']}/20",
                        fmt(summary['episode_summary'][m]['minimum']), fmt(summary['episode_summary'][m]['maximum']),
                        fmt(summary['episode_summary'][m]['mean'])] for m in METRICS]), '',
             '전체 점수는 문맥에 같은 가중치를 부여합니다. 위 20개 에피소드 평균의 단순 평균과는 가중치가 다릅니다. '
             '에피소드 수는 독립 학습 run 수가 아니며, 에피소드별 표 자체는 별도 유의성 검정이 아닙니다.', '',
             '## 단계별 전역 평균 가치', '',
             '각 단계에서 문맥 전체의 예측 가치를 먼저 평균한 17점 프로파일입니다. '
             '이 프로파일과 단계 순서를 상관시킨 값은 문맥별 상관계수의 평균과 다릅니다.', '',
             md_table(['전역 프로파일 지표', '값'], [[m, fmt(summary['global_profile_correlations']['random'][m])] for m in METRICS]), '',
             md_table(['Stage', '평균 critic 값 [episode-bootstrap 95% CI]', 'SD', '중앙값', '최솟값', '최댓값'],
                      [[p['stage'], f"{fmt(p['mean'])} [{fmt(p['ci95_low'])}, {fmt(p['ci95_high'])}]",
                        fmt(p['sd']), fmt(p['median']), fmt(p['minimum']), fmt(p['maximum'])] for p in profiles]), '',
             f"전역 평균 가치가 가장 높은 단계는 {summary['global_profile_peak_stage']['random']}입니다. "
             f"평균 가치는 stage 0의 {fmt(profiles[0]['mean'])}에서 stage 6의 {fmt(profiles[6]['mean'])}까지 하락한 뒤 "
             f"다시 상승하며, stage 15의 {fmt(profiles[15]['mean'])}에서 최대가 되고 stage 16은 {fmt(profiles[16]['mean'])}입니다. "
             '단계별 평균 가치의 CI는 문맥을 에피소드 단위로 재표집한 구간입니다. '
             '전역 프로파일의 상관계수 자체에는 여기서 CI를 부여하지 않았습니다. '
             'critic 값 크기는 true expected return의 정확도를 검증한 결과가 아닙니다.', '',
             '## 저장된 Baseline·FLASH와의 문맥별 비교', '',
             '이 절은 새 모델 평가 없이 기존 seed 3710의 저장 예측과 비교한 보조 분석입니다. '
             '세 모델의 frame_id, episode_id, episode_step, block_id 및 이글루 템플릿이 일치함을 검증했습니다. '
             '문맥은 paired이지만 학습 seed는 matched가 아닙니다(Random 11010, Baseline/FLASH 3710). '
             '모든 차이는 Random − reference입니다.', '',
             md_table(['모델', '평균 ρ [95% CI]', '평균 τ_b [95% CI]'],
                      [[MODEL_LABELS[name]] + [cell(summary['models'][name][m]) for m in METRICS]
                       for name in ('baseline', 'random', 'flash')]), '',
             md_table(['차이', '지표', '평균 차이 [95% CI]', '중앙값', 'Q1', 'Q3', 'IQR', '제외 쌍'],
                      [[c['comparison'], c['metric'], cell(c), fmt(c['median']), fmt(c['q1']), fmt(c['q3']), fmt(c['iqr']), c['undefined']]
                       for c in comparisons]), '',
             md_table(['차이', '지표', 'Random이 더 높음', '동률', '더 낮음', '양수 조건부 평균 차이', '음수 조건부 평균 차이'],
                      [[c['comparison'], c['metric']]
                       + [f"{c[k]:,} ({100*c[k + '_fraction']:.2f}%)" for k in ('positive', 'zero', 'negative')]
                       + [fmt(c['positive_conditional_mean'], True), fmt(c['negative_conditional_mean'], True)] for c in comparisons]), '',
             '차이의 CI는 각 모델의 개별 CI 겹침이 아니라, 동일 에피소드들을 함께 재표집한 문맥별 차이의 평균에서 계산했습니다. '
             '다중 비교 보정이나 별도 p-value는 적용하지 않았습니다.', '',
             f"Random−Baseline의 Spearman 평균 차이는 {cell(baseline_rho)}이지만, "
             f"Random이 더 높은 문맥은 {100*baseline_rho['positive_fraction']:.2f}%이고 차이의 중앙값은 {fmt(baseline_rho['median'])}입니다. "
             f"양수 변화의 조건부 평균 {fmt(baseline_rho['positive_conditional_mean'], True)}이 "
             f"음수 변화의 조건부 평균 {fmt(baseline_rho['negative_conditional_mean'], True)}보다 절대 크기가 커서 "
             '절반 미만의 문맥이 개선되어도 전체 평균 차이는 양수일 수 있습니다. '
             'Baseline 대비 Spearman 차이의 CI는 0보다 높고 Kendall 차이의 CI는 0을 포함합니다. '
             'FLASH 대비 두 지표의 평균 차이는 모두 음수이며 각 차이 CI도 0보다 낮습니다. '
             '이는 평가한 고정 체크포인트들에 대한 문맥 재표집 결과입니다.', '',
             md_table(['Episode', '평가 문맥', 'Random−Baseline ρ', 'Random−Baseline τ_b', 'Random−FLASH ρ', 'Random−FLASH τ_b'],
                      [[e['episode'], e['evaluated_contexts']]
                       + [fmt(e[f'{ref}_{m}_difference_mean'], True) for ref in ('baseline', 'flash') for m in METRICS]
                       for e in comparison_episodes]), '',
             md_table(['참조 모델', '지표', '양수 episode 평균 차이', '음수 episode 평균 차이', '최솟값', '최댓값'],
                      [[ref.upper() if ref == 'flash' else 'Baseline', m,
                        summary['reference_episode_summary'][ref][m]['positive'],
                        summary['reference_episode_summary'][ref][m]['negative'],
                        fmt(summary['reference_episode_summary'][ref][m]['minimum'], True),
                        fmt(summary['reference_episode_summary'][ref][m]['maximum'], True)]
                       for ref in ('baseline', 'flash') for m in METRICS]), '',
             '## 해석 범위', '',
             'Random seed 11010의 평균 정렬 점수는 두 지표 모두 양수지만 절대 크기는 작고 문맥별 분포가 넓습니다. '
             '평균의 순서는 이 체크포인트들에서 Baseline < Random < FLASH입니다. '
             '이 순서는 모든 문맥이나 에피소드에서의 우열을 뜻하지 않습니다.', '',
             'CI는 고정 체크포인트에 대한 평가 문맥의 불확실성만 반영하며 학습 seed 간 변동과 반복 latent draw의 변동은 포함하지 않습니다. '
             'Random 체크포인트는 최종 평가 평균 1,000점 초과라는 조건으로 선택되었으므로 전체 Random variant의 대표 표본으로 가정하지 않습니다.', '',
             'stage 증가에 대한 critic 반응을 측정하는 시각 개입이며, 모든 배경에서 true expected return이 단조 증가한다는 가정의 검증이나 '
             'latent disentanglement·spatial entanglement 해소의 직접 증거는 아닙니다. '
             '패치한 관측·행동 이력이 실제 환경에서 실현 가능한 trajectory인지도 보장하지 않습니다. '
             '배경 노이즈 MLP probe 실험과는 별도 분석입니다.', '',
             '## 파일 안내', '',
             md_table(['파일', '내용'], [
                 ['appendix_random_seed11010.tex', '논문에 옮겨 쓸 영문 설명·수식·LaTeX 표'],
                 ['statistics_detailed.json', '반올림 전 전체 통계와 평가·분석 프로토콜'],
                 ['per_context_details.csv', '17,421개 문맥의 메타데이터·17개 가치·정렬 점수·참조 모델 점수·차이'],
                 ['per_stage_values_detailed.csv', '296,157개 문맥×단계 가치와 문맥 ID'],
                 ['per_episode.csv / per_block.csv', '20개 에피소드 / 320개 블록별 결과'],
                 ['per_stage_profile.csv', '17단계 전역 가치 분포와 CI 및 참조 모델 평균'],
                 ['stage_transition_statistics.csv', '인접 단계 및 0→16의 문맥별 가치 차이 통계'],
                 ['original_stage_counts.csv', '원래 RAM 단계별 수집·평가 문맥 수'],
                 ['reference_comparisons.csv', 'Random−Baseline / Random−FLASH의 상세 분포와 paired CI'],
                 ['reference_comparisons_per_episode.csv', '에피소드별 두 참조 모델과의 차이'],
                 ['bootstrap_replicates.npz', '10,000회 episode 선택·분모·정렬 평균·paired 차이·단계별 가치 평균'],
                 ['policy_evaluation_returns.csv', '별도의 최종 BF16 정책 평가 20개 return'],
                 ['figures/', '분포·에피소드·가치 프로파일·참조 차이의 PNG/PDF'],
                 ['raw_snapshot/', '분석 입력 NPZ·프로토콜·평가 기록·당시 내보낸 코드 사본'],
                 ['manifest.json', '입력·결과 SHA-256, 소프트웨어 버전, 재현 명령과 검증 기록']]), '',
             '![문맥별 정렬 분포](figures/context_distributions.png)', '',
             '![에피소드별 평균 정렬](figures/episode_means.png)', '',
             '![전역 가치 프로파일](figures/global_profiles.png)', '',
             '![참조 모델과 문맥별 차이](figures/reference_differences.png)', '']
    (output / 'README_ko.md').write_text('\n'.join(lines), encoding='utf-8')


def make_latex(output, summary, episodes, profiles, comparisons, comparison_episodes):
    stats = summary['models']['random']
    baseline_rho = next(c for c in comparisons if c['reference'] == 'baseline' and c['metric'] == 'spearman')
    content = [r'% Generated from saved predictions. Requires booktabs; no manuscript file was edited.',
               r'\subsection{Controlled Igloo-Stage Evaluation of Random Neighbors}',
               r'\label{app:random11010_alignment}',
               r'\paragraph{Checkpoint and shared contexts.}',
               'We evaluate the fixed world model and critic of the Random-neighbors variant trained with seed 11010, '
               'using checkpoints saved at 100,000 environment steps. This variant retains the multiplicative '
               'value-difference/TD-error anchor rule and replaces non-anchor neighbor contexts with random replay contexts. '
               'Its separate final BF16 policy evaluation averages 2177.0 return over 20 episodes; these policy-evaluation '
               'episodes are distinct from the critic-diagnostic episodes below. '
               'The checkpoint was selected under a final-return-above-1000 criterion, rather than sampled uniformly from training runs.',
               '',
               'We reuse the 20,000 agent-step observations collected by the Baseline policy trained with seed 3710, '
               'at action repeat 4 and image resolution $64\\times64$. The rollout comprises 19 complete episodes '
               'and part of a twentieth episode. The Random policy does not collect new contexts for this analysis. '
               'Consecutive observations with unchanged RAM-derived igloo stage are split into blocks of at most 64 '
               'observations, without crossing episode boundaries. Blocks shorter than nine observations are excluded. '
               'For each of the 320 valid blocks, we replace the igloo region throughout the block with each of the '
               '17 stage templates ($0$--$16$), keeping other pixels and all actions fixed across stage variants. '
               'No background noise is injected. The first eight observations initialize temporal context; '
               'the remaining time points yield 17,421 evaluated contexts and 296,157 context-stage predictions. '
               'The recorded sampling protocol resets the random seed per block and stage variant. Each prediction '
               'uses one latent draw, without averaging repeated draws.', '',
               r'\paragraph{Per-context alignment and uncertainty.}',
               r'For each context $i$, we calculate '
               r'$\rho_i=\operatorname{Spearman}((0,\ldots,16),(v_i(0),\ldots,v_i(16)))$ '
               r"and analogously Kendall's $\tau_b$. We average these coefficients with equal weight per context. "
               'Standard tie handling is used. Constant predictions would yield undefined correlations and be '
               'excluded rather than assigned zero; none occur here. '
               'We use 10,000 whole-episode bootstrap replicates with seed 2027. Each replicate samples 20 episode '
               'identifiers with replacement, includes all their evaluated contexts with the sampled multiplicity, '
               'and recomputes the context-count denominator. The 2.5th and 97.5th percentiles define 95\\% confidence intervals. '
               'These intervals describe evaluation-context uncertainty for a fixed checkpoint and exclude variation '
               'across training seeds and repeated latent draws.', '',
               latex_table('metrics', 'Mean per-context stage--value correlations for Random neighbors, training seed 11010. '
                           'Brackets denote episode-bootstrap percentile 95\\% CIs.',
                           ['Metric', 'Mean [95\\% CI]', 'Defined contexts', 'Undefined'],
                           [[r"Spearman's $\rho$" if m == 'spearman' else r"Kendall's $\tau_b$",
                             cell(stats[m]), stats[m]['n'], stats[m]['undefined']] for m in METRICS]),
               r'\paragraph{Distributions across contexts.}',
               'The positive means coexist with broad context-level distributions. Quartiles measure dispersion '
               'across contexts, whereas confidence intervals quantify uncertainty in a mean. '
               'The standard deviations below use $N-1$ as their denominator.', '',
               latex_table('distribution', 'Distribution of per-context correlations across all 17,421 evaluated contexts.',
                           ['Metric', 'SD', 'Median', 'Q1', 'Q3', 'IQR'],
                           [[r'$\rho$' if m == 'spearman' else r'$\tau_b$']
                            + [fmt(stats[m][k]) for k in ('sd', 'median', 'q1', 'q3', 'iqr')] for m in METRICS]),
               latex_table('direction', 'Signs and conditional magnitudes of per-context correlations. '
                           'These are absolute correlations, not changes relative to another model.',
                           ['Statistic', r"Spearman's $\rho$", r"Kendall's $\tau_b$"],
                           [[title] + [f"{stats[m][key]:,} ({100*stats[m][key + '_fraction']:.2f}\\%)" for m in METRICS]
                            for title, key in [('Positive', 'positive'), ('Zero', 'zero'), ('Negative', 'negative')]]
                           + [[title] + [fmt(stats[m][key]) for m in METRICS]
                              for title, key in [('Mean where positive', 'positive_conditional_mean'),
                                                 ('Mean where negative', 'negative_conditional_mean')]]),
               r'\paragraph{Variation across evaluation episodes.}',
               'We also average correlations separately within each diagnostic episode. The final episode is partial. '
               'These are descriptive summaries for one checkpoint, not independent training-seed replications. '
               'The pooled estimates weight contexts equally and need not equal the unweighted average of episode means. '
               f"Episode-mean Spearman correlation is positive in {summary['episode_summary']['spearman']['positive']} of 20 episodes "
               f"and episode-mean Kendall correlation is positive in {summary['episode_summary']['kendall']['positive']} of 20 episodes.", '',
               latex_table('episodes', 'Mean Random-neighbors correlations by diagnostic episode. The final episode is partial.',
                           ['Episode', 'Evaluated contexts', r'Mean $\rho$', r'Mean $\tau_b$'],
                           [[e['episode'], f"{e['evaluated_contexts']:,}", fmt(e['spearman_mean']), fmt(e['kendall_mean'])]
                            for e in episodes]),
               r'\paragraph{Global average value profile.}',
               r'We additionally form $\bar v(k)=N^{-1}\sum_i v_i(k)$ and correlate this 17-point '
               'profile with stage index. This reverses the order of averaging and correlation relative to the '
               f"per-context metric. The resulting Spearman and Kendall coefficients are "
               f"{fmt(summary['global_profile_correlations']['random']['spearman'])} and "
               f"{fmt(summary['global_profile_correlations']['random']['kendall'])}, respectively. "
               f"The profile reaches its maximum at stage {summary['global_profile_peak_stage']['random']}. "
               'These profile correlations are point estimates. The intervals in the following table refer '
               'to mean predicted values, not to profile correlations.', '',
               latex_table('profile', 'Global mean predicted value by igloo stage for Random neighbors. '
                           'Brackets give whole-episode bootstrap percentile 95\\% CIs for mean values.',
                           ['Stage', 'Mean value [95\\% CI]', 'SD', 'Median'],
                           [[p['stage'], f"{fmt(p['mean'])} [{fmt(p['ci95_low'])}, {fmt(p['ci95_high'])}]",
                             fmt(p['sd']), fmt(p['median'])] for p in profiles]),
               r'\paragraph{Supplementary comparisons with saved Baseline and FLASH predictions.}',
               'For reference, we compare with the previously saved Baseline and FLASH checkpoints trained with '
               'seed 3710. Frame identifiers, episode identifiers, within-episode steps, block identifiers, and '
               'stage templates match exactly. Comparisons are paired over evaluation contexts, while the training '
               'seed differs between Random and the reference models. All differences below are Random minus reference. '
               'We reuse the same episode-bootstrap selections for both sides, apply no multiplicity correction, '
               'and do not report separate p-values.', '',
               latex_table('comparison_means', 'Saved checkpoint comparison on identical contexts. '
                           'Training seeds differ between Random and the reference models. Brackets give 95\\% CIs.',
                           ['Model', r"Spearman's $\rho$", r"Kendall's $\tau_b$"],
                           [[MODEL_LABELS[name]] + [cell(summary['models'][name][m]) for m in METRICS]
                            for name in ('baseline', 'random', 'flash')]),
               latex_table('differences', 'Paired context-weighted mean differences, Random minus reference. '
                           'Intervals quantify fixed-checkpoint context uncertainty.',
                           ['Difference', 'Metric', 'Mean [95\\% CI]', 'Median', 'Q1', 'Q3'],
                           [[c['comparison'], r'$\rho$' if c['metric'] == 'spearman' else r'$\tau_b$',
                             cell(c), fmt(c['median']), fmt(c['q1']), fmt(c['q3'])] for c in comparisons]),
               latex_table('changes', 'Frequency and conditional magnitude of matched-context changes. '
                           'All changes are Random minus reference.',
                           ['Reference', 'Metric', 'Higher', 'Lower', 'Equal', 'Positive mean', 'Negative mean'],
                           [[c['reference'].capitalize(), r'$\rho$' if c['metric'] == 'spearman' else r'$\tau_b$']
                            + [f"{c[k]:,}" for k in ('positive', 'negative', 'zero')]
                            + [fmt(c['positive_conditional_mean'], True), fmt(c['negative_conditional_mean'], True)]
                            for c in comparisons]),
               f"Relative to Baseline, the mean Spearman difference is {fmt(baseline_rho['mean'], True)}, "
               f"while its median is {fmt(baseline_rho['median'], True)} and "
               f"only {100*baseline_rho['positive_fraction']:.2f}\\% of contexts have a positive difference. "
               f"The conditional positive mean ({fmt(baseline_rho['positive_conditional_mean'], True)}) has a larger "
               f"absolute magnitude than the conditional negative mean ({fmt(baseline_rho['negative_conditional_mean'], True)}). "
               'Thus a positive mean difference coexists with improvement in fewer than half of matched contexts. '
               'The episode-bootstrap difference interval lies above zero for Spearman relative to Baseline, '
               'whereas the Kendall interval includes zero. Relative to FLASH, both mean differences and their '
               'intervals lie below zero. These statements concern the evaluated fixed checkpoints, not '
               'across-seed variant effects.', '',
               latex_table('episode_differences', 'Episode-specific paired mean differences: Random minus each reference. '
                           'The final diagnostic episode is partially observed.',
                           ['Episode', 'Contexts', r'$\Delta\rho_{R-B}$', r'$\Delta\tau_{R-B}$',
                            r'$\Delta\rho_{R-F}$', r'$\Delta\tau_{R-F}$'],
                           [[e['episode'], f"{e['evaluated_contexts']:,}"]
                            + [fmt(e[f'{ref}_{m}_difference_mean'], True) for ref in ('baseline', 'flash') for m in METRICS]
                            for e in comparison_episodes]),
               'The Random-minus-Baseline episode-mean difference is positive in 12 of 20 episodes for Spearman '
               'and 11 of 20 for Kendall. The corresponding counts against FLASH are 5 of 20 and 1 of 20. '
               'These counts are descriptive, without separate episode-level significance tests.', '',
               r'\paragraph{Interpretation and scope.}',
               'For these fixed checkpoints, mean per-context alignment follows Baseline $<$ Random $<$ FLASH. '
               'Absolute correlations remain weak on average and vary substantially across contexts. '
               'This diagnostic measures critic responses to controlled visual stage changes; it does not establish '
               'that true policy-dependent expected return is monotonic in stage in every context, that edited '
               'observation--action histories are realizable trajectories, or that latent representations are '
               'disentangled. The results are separate from the background-noise classification probe. '
               'Differences between these checkpoints do not estimate an across-training-seed variant effect.', '']
    (output / 'appendix_random_seed11010.tex').write_text('\n'.join(content), encoding='utf-8')


def figures(output, scores, profiles, episodes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = output / 'figures'
    directory.mkdir()

    def save(fig, name):
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(directory / f'{name}.{extension}', dpi=180)
        plt.close(fig)

    colors = {'random': '#b85a24', 'baseline': '#596d82', 'flash': '#2f8768'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for j, metric in enumerate(METRICS):
        for name in ('baseline', 'random', 'flash'):
            x = scores[name][:, j]
            axes[j].hist(x[np.isfinite(x)], bins=np.linspace(-1, 1, 31), histtype='step',
                         color=colors[name], label=MODEL_LABELS[name], linewidth=1.5)
        axes[j].set(xlabel=metric, ylabel='Contexts', title='Per-context alignment')
    axes[0].legend(fontsize=8)
    save(fig, 'context_distributions')

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for j, metric in enumerate(METRICS):
        for name in ('baseline', 'random', 'flash'):
            means = [float(scores[name][e['episode_mask'], j].mean()) for e in episodes]
            axes[j].plot(range(1, 21), means, marker='o', ms=3, color=colors[name], label=MODEL_LABELS[name])
        axes[j].axhline(0, color='black', linewidth=.6)
        axes[j].set(ylabel=f'Mean {metric}')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Diagnostic episode (20 is partial)', xticks=range(1, 21))
    save(fig, 'episode_means')

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(17)
    for name in ('baseline', 'random', 'flash'):
        ax.plot(x, [p[f'{name}_mean'] for p in profiles], marker='o', ms=3,
                color=colors[name], label=MODEL_LABELS[name])
    ax.fill_between(x, [p['ci95_low'] for p in profiles], [p['ci95_high'] for p in profiles],
                    color=colors['random'], alpha=.18, label='Random mean-value 95% CI')
    ax.set(xlabel='Igloo stage', ylabel='Mean predicted value', xticks=x)
    ax.legend(fontsize=8)
    save(fig, 'global_profiles')

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for row, reference in enumerate(('baseline', 'flash')):
        for j, metric in enumerate(METRICS):
            delta = scores['random'][:, j] - scores[reference][:, j]
            axes[row, j].hist(delta[np.isfinite(delta)], bins=np.linspace(-2, 2, 41), color=colors['random'])
            axes[row, j].axvline(0, color='black', linewidth=.8)
            axes[row, j].set(xlabel=f'Random minus {reference}: {metric}', ylabel='Contexts')
    save(fig, 'reference_differences')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--contexts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory; detailed exports are never overwritten')
    result, reference, output = args.result.resolve(), args.reference.resolve(), args.output.resolve()
    protocol = json.loads((result / 'protocol.json').read_text())
    if protocol['run'] != 'Frostbite-score-screen-frostbite_neighbor_screen_20261107-seed11010_randomneighor':
        parser.error('This appendix export targets Random-neighbors seed 11010')
    if protocol['bootstrap_unit'] != 'episode':
        parser.error('This appendix export requires whole-episode bootstrap')
    raw = read_npz(result / 'values.npz')
    metadata = {k: raw[k] for k in META}
    values = {'random': raw['values'], **match_reference(read_npz(reference / 'paired_values.npz'), metadata)}
    contexts = read_npz(args.contexts)
    for key in ('episode_id', 'episode_step'):
        np.testing.assert_array_equal(metadata[key], contexts[key][metadata['frame_id']])
    np.testing.assert_array_equal(read_npz(result / 'used_templates.npz')['templates'],
                                  read_npz(reference / 'used_templates.npz')['templates'])
    ref_protocol = json.loads((reference / 'protocol.json').read_text())
    for key in ('collection_run', 'collection_step', 'collection_seed', 'context_warmup', 'max_block_length', 'seed'):
        if protocol[key] != ref_protocol[key]:
            raise ValueError(f'Different reference protocol: {key}')
    stages = np.where(contexts['ram'][:, 77] == 255, 0, contexts['ram'][:, 77].astype(int) + 1)
    blocks = make_blocks(stages, contexts['episode_id'], protocol['max_block_length'], protocol['context_warmup'])
    frame_ids = np.concatenate([np.arange(start + 8, end) for start, end in blocks])
    block_ids = np.concatenate([np.full(end - start - 8, i) for i, (start, end) in enumerate(blocks)])
    np.testing.assert_array_equal(metadata['frame_id'], frame_ids)
    np.testing.assert_array_equal(metadata['block_id'], block_ids)
    scores = {}
    for name, array in values.items():
        if array.shape != (len(frame_ids), 17) or not np.isfinite(array).all():
            raise ValueError(f'Invalid predictions: {name}')
        scores[name] = correlations(array)
        print(f'Correlations recomputed: {name}, {len(array):,} contexts', flush=True)
    ids = np.unique(metadata['episode_id'])
    repetitions, seed = protocol['bootstrap'], protocol['seed']
    selections = np.random.default_rng(seed).integers(0, len(ids), size=(repetitions, len(ids)))
    draws = {}
    model_stats = {}
    for name in values:
        boot = resampled_means(scores[name], metadata['episode_id'], selections)
        model_stats[name] = {}
        for j, metric in enumerate(METRICS):
            draws[f'{name}_{metric}'] = boot[:, j]
            model_stats[name][metric] = dict(describe(scores[name][:, j]), ci95=interval(boot[:, j]))
    saved_stats = json.loads((result / 'statistics.json').read_text())
    saved_ref = json.loads((reference / 'statistics.json').read_text())
    for name in values:
        for metric in METRICS:
            expected_mean = saved_stats['metrics'][metric]['mean'] if name == 'random' else saved_ref['metrics'][metric]['means'][0 if name == 'baseline' else 1]
            expected_ci = saved_stats['metrics'][metric]['ci95'] if name == 'random' else saved_ref['metrics'][metric]['ci95'][0 if name == 'baseline' else 1]
            np.testing.assert_allclose(model_stats[name][metric]['mean'], expected_mean, rtol=0, atol=1e-12)
            np.testing.assert_allclose(model_stats[name][metric]['ci95'], expected_ci, rtol=0, atol=1e-12)
    print('Saved means and CIs reproduced for all three checkpoints', flush=True)
    comparisons = []
    for ref in ('baseline', 'flash'):
        delta = scores['random'] - scores[ref]
        boot = resampled_means(delta, metadata['episode_id'], selections)
        for j, metric in enumerate(METRICS):
            draws[f'random_minus_{ref}_{metric}'] = boot[:, j]
            ci = interval(boot[:, j])
            comparisons.append(dict(comparison=f'Random - {ref.capitalize()}', reference=ref, metric=metric,
                                    **describe(delta[:, j]), ci95=ci, ci_excludes_zero=bool(ci[0] > 0 or ci[1] < 0)))
    stage_draws = resampled_means(values['random'], metadata['episode_id'], selections)
    draws['random_stage_mean_values'] = stage_draws
    profiles = []
    for stage in range(17):
        stat = describe(values['random'][:, stage])
        low, high = interval(stage_draws[:, stage])
        profiles.append(dict(stage=stage, **stat, ci95_low=low, ci95_high=high,
                             **{name + '_mean': float(v[:, stage].mean()) for name, v in values.items()}))
    transitions = []
    for start, end in [(k, k + 1) for k in range(16)] + [(0, 16)]:
        delta = values['random'][:, end] - values['random'][:, start]
        boot = resampled_means(delta, metadata['episode_id'], selections)[:, 0]
        low, high = interval(boot)
        transitions.append(dict(stage_from=start, stage_to=end, **describe(delta), ci95_low=low, ci95_high=high))
        draws[f'random_value_stage_{end}_minus_{start}'] = boot
    episodes, comparison_episodes, plot_episodes = [], [], []
    for eid in ids:
        mask = metadata['episode_id'] == eid
        collection_mask = contexts['episode_id'] == eid
        row = dict(episode_id=int(eid), episode=int(eid) + 1,
                   collected_frames=int(collection_mask.sum()), evaluated_contexts=int(mask.sum()),
                   blocks=int(len(np.unique(block_ids[mask]))), completed=bool(contexts['episode_end'][collection_mask].any()),
                   first_frame_id=int(frame_ids[mask].min()), last_frame_id=int(frame_ids[mask].max()))
        for j, metric in enumerate(METRICS):
            row.update({metric + '_' + k: v for k, v in describe(scores['random'][mask, j]).items()})
        episodes.append(row)
        plot_episodes.append(dict(row, episode_mask=mask))
        pair_row = {k: row[k] for k in ('episode_id', 'episode', 'evaluated_contexts')}
        for ref in ('baseline', 'flash'):
            for j, metric in enumerate(METRICS):
                pair_row[f'{ref}_{metric}_difference_mean'] = float((scores['random'][mask, j] - scores[ref][mask, j]).mean())
        comparison_episodes.append(pair_row)
    block_rows = []
    for bid, (start, end) in enumerate(blocks):
        mask = block_ids == bid
        row = dict(block_id=bid, episode_id=int(contexts['episode_id'][start]), start_frame_id=start,
                   end_frame_id_exclusive=end, original_stage=int(stages[start]),
                   collected_frames=end - start, warmup_frames=8, evaluated_contexts=int(mask.sum()))
        for j, metric in enumerate(METRICS):
            row.update({metric + '_' + k: v for k, v in describe(scores['random'][mask, j]).items()})
        block_rows.append(row)
    candidates, position = [], 0
    while position < len(stages):
        end = position + 1
        while (end < len(stages) and end - position < 64 and stages[end] == stages[position]
               and contexts['episode_id'][end] == contexts['episode_id'][position]):
            end += 1
        candidates.append((position, end))
        position = end
    short = [(start, end) for start, end in candidates if end - start < 9]
    accounting = dict(collected_frames=len(stages), candidate_blocks=len(candidates),
                      excluded_short_blocks=len(short), excluded_short_frames=sum(end-start for start, end in short),
                      valid_blocks=len(blocks), warmup_frames=8 * len(blocks), evaluated_contexts=len(frame_ids),
                      context_stage_predictions=int(values['random'].size))
    assert accounting['collected_frames'] == sum(accounting[k] for k in ('excluded_short_frames', 'warmup_frames', 'evaluated_contexts'))
    root = Path(__file__).resolve().parents[1]
    final_path = root / 'runs' / protocol['run'] / 'final_evaluation.json'
    final_evaluation = json.loads(final_path.read_text())
    if final_evaluation['seed'] != 11010 or final_evaluation['checkpoint_step'] != protocol['step']:
        raise ValueError('Unexpected final policy evaluation identity')
    with (root / 'results/wandb_runs_classification.csv').open() as stream:
        classification = next(row for row in csv.DictReader(stream) if row.get('Local Run Name') == protocol['run'])
    if classification['Neighbor Source'] != 'random' or classification['Score Combination'] != 'multiply':
        raise ValueError('Unexpected training variant')
    np.testing.assert_allclose(np.mean(final_evaluation['episode_returns']), final_evaluation['episode_avg_return'])
    summary = dict(protocol=protocol, training_metadata=classification, context_accounting=accounting,
                   analysis=dict(repetitions=repetitions, seed=seed, unit='episode',
                                 interval='percentile 95%', sign_counts='Exact unrounded signs; zero is exact equality',
                                 descriptive_sd='sample standard deviation, ddof=1',
                                 comparison_direction='Random minus reference',
                                 training_seeds_matched=False, inference_rerun=False),
                   models=model_stats, reference_comparisons=comparisons,
                   episode_summary={m: describe([row[m + '_mean'] for row in episodes]) for m in METRICS},
                   reference_episode_summary={ref: {m: describe([row[f'{ref}_{m}_difference_mean'] for row in comparison_episodes])
                                                   for m in METRICS} for ref in ('baseline', 'flash')},
                   global_profile_correlations={name: {m: float(value) for m, value in zip(METRICS, correlations(v.mean(axis=0)[None])[0])}
                                                for name, v in values.items()},
                   global_profile_peak_stage={name: int(v.mean(axis=0).argmax()) for name, v in values.items()},
                   final_policy_evaluation=final_evaluation,
                   final_policy_return_distribution=describe(final_evaluation['episode_returns']))
    output.mkdir(parents=True)
    save_json(output / 'statistics_detailed.json', summary)
    save_csv(output / 'per_episode.csv', episodes)
    save_csv(output / 'per_block.csv', block_rows)
    save_csv(output / 'per_stage_profile.csv', profiles)
    save_csv(output / 'stage_transition_statistics.csv', transitions)
    save_csv(output / 'reference_comparisons.csv', comparisons)
    save_csv(output / 'reference_comparisons_per_episode.csv', comparison_episodes)
    save_csv(output / 'original_stage_counts.csv', [dict(stage=k, collected_frames=int((stages == k).sum()),
                                                       evaluated_contexts=int((stages[frame_ids] == k).sum())) for k in range(17)])
    save_csv(output / 'policy_evaluation_returns.csv', [dict(policy_eval_episode=i + 1, episode_return=value,
                                                          checkpoint_step=100000, precision=final_evaluation['precision'])
                                                     for i, value in enumerate(final_evaluation['episode_returns'])])
    def per_context_rows():
        for i, frame in enumerate(frame_ids):
            bid = block_ids[i]
            row = {key: int(metadata[key][i]) for key in META}
            row.update(episode=int(metadata['episode_id'][i]) + 1, original_stage=int(stages[frame]),
                       block_start_frame_id=blocks[bid][0], history_length=int(frame - blocks[bid][0]),
                       unique_stage_values=int(len(np.unique(values['random'][i]))),
                       argmax_stage_first=int(values['random'][i].argmax()), argmin_stage_first=int(values['random'][i].argmin()),
                       value_range=float(np.ptp(values['random'][i])),
                       value_stage16_minus_stage0=float(values['random'][i, 16] - values['random'][i, 0]))
            for name in values:
                row.update({name + '_' + m: float(scores[name][i, j]) for j, m in enumerate(METRICS)})
            for ref in ('baseline', 'flash'):
                row.update({f'random_minus_{ref}_{m}': float(scores['random'][i, j] - scores[ref][i, j]) for j, m in enumerate(METRICS)})
            row.update({f'value_stage_{k:02d}': float(v) for k, v in enumerate(values['random'][i])})
            yield row
    save_csv(output / 'per_context_details.csv', per_context_rows())
    save_csv(output / 'per_stage_values_detailed.csv',
             (dict(frame_id=int(frame), episode_id=int(metadata['episode_id'][i]),
                   episode_step=int(metadata['episode_step'][i]), block_id=int(block_ids[i]),
                   stage=k, value=float(v)) for i, frame in enumerate(frame_ids) for k, v in enumerate(values['random'][i])))
    group_counts = np.array([(metadata['episode_id'] == eid).sum() for eid in ids])
    np.savez_compressed(output / 'bootstrap_replicates.npz', episode_ids=ids,
                        selected_episode_indices=selections.astype(np.int16),
                        selected_context_counts=group_counts[selections].sum(axis=1), **draws)
    make_report(output, summary, episodes, profiles, comparisons, comparison_episodes)
    make_latex(output, summary, episodes, profiles, comparisons, comparison_episodes)
    figures(output, scores, profiles, plot_episodes)
    snapshot = output / 'raw_snapshot'
    inputs = [('random_values.npz', result / 'values.npz'), ('random_protocol.json', result / 'protocol.json'),
              ('random_statistics.json', result / 'statistics.json'), ('used_templates.npz', result / 'used_templates.npz'),
              ('reference_paired_values.npz', reference / 'paired_values.npz'),
              ('reference_protocol.json', reference / 'protocol.json'), ('reference_statistics.json', reference / 'statistics.json'),
              ('shared_contexts.npz', args.contexts.resolve()), ('final_evaluation.json', final_path),
              ('eval_frostbite_value.py', Path(__file__).with_name('eval_frostbite_value.py')),
              ('frostbite_value_stats.py', Path(__file__).with_name('frostbite_value_stats.py')),
              ('export_frostbite_alignment_details.py', Path(__file__).resolve()),
              ('iclr2027_conference_at_export.tex', root.parent / 'iclr2027_conference.tex')]
    snapshot.mkdir()
    manifest_inputs = []
    for name, source in inputs:
        destination = snapshot / name
        shutil.copy2(source, destination)
        manifest_inputs.append(dict(source=str(source), snapshot=str(destination.relative_to(output)),
                                    bytes=source.stat().st_size, sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    import scipy
    import matplotlib
    manifest = dict(inputs=manifest_inputs,
                    software=dict(numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
                    repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                    source_snapshot_scope='Evaluator source as present at export time; uncommitted changes are preserved in the snapshot.',
                    verification=['All per-context correlations recomputed from saved 17-stage predictions',
                                  'All three saved model means and CIs reproduced to absolute tolerance 1e-12',
                                  'Context IDs, episode IDs, within-episode steps, block IDs and templates match exactly',
                                  'Frame accounting and final policy evaluation average verified'],
                    reproduction_command=' '.join(['python probing/export_frostbite_alignment_details.py',
                                                   '--result', str(result), '--reference', str(reference),
                                                   '--contexts', str(args.contexts.resolve()), '--output', str(output) + '_reproduced']))
    manifest['outputs'] = [dict(path=str(p.relative_to(output)), bytes=p.stat().st_size,
                                sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                           for p in sorted(output.rglob('*')) if p.is_file() and snapshot not in p.parents]
    save_json(output / 'manifest.json', manifest)
    archive = shutil.make_archive(str(output), 'zip', root_dir=output)
    print(json.dumps(dict(output=str(output), archive=archive, models=model_stats,
                          reference_comparisons=comparisons, context_accounting=accounting), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
