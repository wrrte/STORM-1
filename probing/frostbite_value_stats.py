"""CPU-only paired context statistics for eval_frostbite_value.py."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau


def make_blocks(stages, episode_ids, max_length=64, warmup=8):
    """Never carry transformer context over an episode boundary."""
    blocks, i = [], 0
    while i < len(stages):
        j = i + 1
        while (j < len(stages) and j - i < max_length
               and stages[j] == stages[i] and episode_ids[j] == episode_ids[i]):
            j += 1
        if j - i > warmup:
            blocks.append((i, j))
        i = j
    return blocks


def correlations(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 17:
        raise ValueError("Expected shape (frames, 17)")
    result = np.full((len(values), 2), np.nan)
    for i, row in enumerate(values):
        if np.isfinite(row).all() and np.ptp(row) > 0:
            result[i] = spearmanr(np.arange(17), row)[0], kendalltau(np.arange(17), row)[0]
    return result


def paired_bootstrap(baseline, flash, groups, repetitions=10000, seed=2027):
    """Paired cluster percentile CI for frame-weighted means and their difference.

    Resample whole episodes (or explicitly selected blocks), retaining all their
    valid matched frames. Recompute the denominator for unequal cluster sizes.
    """
    baseline, flash, groups = map(np.asarray, (baseline, flash, groups))
    valid = np.isfinite(baseline) & np.isfinite(flash)
    baseline, flash, groups = baseline[valid], flash[valid], groups[valid]
    ids, inverse = np.unique(groups, return_inverse=True)
    k = len(ids)
    if not len(baseline):
        return dict(n=0, clusters=0, means=None, ci95=None)
    means = [float(baseline.mean()), float(flash.mean()), float((flash-baseline).mean())]
    if k < 2:
        return dict(n=len(baseline), clusters=k, means=means, ci95=None)
    counts = np.bincount(inverse)
    sums = np.column_stack([np.bincount(inverse, weights=x) for x in (baseline, flash)])
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, 3))
    for i in range(repetitions):
        selected = rng.integers(0, k, k)
        pair = sums[selected].sum(axis=0) / counts[selected].sum()
        draws[i] = pair[0], pair[1], pair[1] - pair[0]
    return dict(n=len(baseline), clusters=k, means=means,
                ci95=np.quantile(draws, [.025, .975], axis=0).T.tolist())


def distribution(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    q1, median, q3 = np.quantile(x, [.25, .5, .75])
    return dict(mean=float(x.mean()), median=float(median), q1=float(q1),
                q3=float(q3), iqr=float(q3-q1))


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_values(output, repetitions=10000, seed=2027, unit='episode'):
    output = Path(output)
    with np.load(output / 'paired_values.npz', allow_pickle=False) as data:
        baseline, flash = data['baseline'], data['flash']
        metadata = {k: data[k] for k in ('frame_id', 'episode_id', 'episode_step', 'block_id')}
    if baseline.shape != flash.shape or len(baseline) == 0:
        raise ValueError('Need nonempty matched value arrays')
    if not np.isfinite(baseline).all() or not np.isfinite(flash).all():
        raise ValueError('Nonfinite value predictions: inspect paired_values.npz before analysis')
    b, f = correlations(baseline), correlations(flash)
    rows, long_rows = [], []
    for i in range(len(b)):
        row = {k: int(v[i]) for k, v in metadata.items()}
        for j, metric in enumerate(('spearman', 'kendall')):
            for name, value in [('baseline', b[i, j]), ('flash', f[i, j]), ('difference', f[i, j]-b[i, j])]:
                row[f'{name}_{metric}'] = float(value) if np.isfinite(value) else ''
        rows.append(row)
        for stage in range(17):
            long_rows.append(dict(frame_id=int(metadata['frame_id'][i]), stage=stage,
                                  baseline_value=float(baseline[i, stage]), flash_value=float(flash[i, stage])))
    write_csv(output / 'per_context_correlations.csv', rows)
    write_csv(output / 'per_stage_values.csv', long_rows)
    result = dict(bootstrap_unit=unit, repetitions=repetitions, seed=seed,
                  interval_method='paired cluster percentile 95%',
                  estimand='Frame-weighted mean over contexts where both correlations are defined',
                  undefined_handling='Constant predictions are undefined; exclude pair for that metric; report counts',
                  uncertainty_scope='Context sampling for two fixed checkpoints only',
                  evaluated_frames=len(b), evaluated_episodes=len(np.unique(metadata['episode_id'])),
                  evaluated_blocks=len(np.unique(metadata['block_id'])), metrics={})
    for j, metric in enumerate(('spearman', 'kendall')):
        valid = np.isfinite(b[:, j]) & np.isfinite(f[:, j])
        delta = f[valid, j] - b[valid, j]
        stats = paired_bootstrap(b[:, j], f[:, j], metadata[f'{unit}_id'], repetitions, seed)
        stats.update(undefined_baseline=int(np.count_nonzero(~np.isfinite(b[:, j]))),
                     undefined_flash=int(np.count_nonzero(~np.isfinite(f[:, j]))),
                     excluded_pairs=int(np.count_nonzero(~valid)),
                     baseline=distribution(b[valid, j]), flash=distribution(f[valid, j]),
                     difference=distribution(delta),
                     improved_fraction=float(np.mean(delta > 0)) if len(delta) else None,
                     tied_fraction=float(np.mean(delta == 0)) if len(delta) else None,
                     worsened_fraction=float(np.mean(delta < 0)) if len(delta) else None,
                     # These reproduce the original separately filtered point estimates.
                     baseline_all_defined=distribution(b[:, j]), flash_all_defined=distribution(f[:, j]))
        ci = stats['ci95']
        stats['difference_ci_excludes_zero'] = bool(ci[2][0] > 0 or ci[2][1] < 0) if ci else None
        result['metrics'][metric] = stats
    profiles = np.column_stack([baseline.mean(axis=0), flash.mean(axis=0)])
    global_corr = correlations(profiles.T)
    result['global_profile_correlations'] = {
        name: {metric: float(global_corr[i, j]) if np.isfinite(global_corr[i, j]) else None
               for j, metric in enumerate(('spearman', 'kendall'))}
        for i, name in enumerate(('baseline', 'flash'))}
    write_csv(output / 'global_profile.csv', [dict(stage=i, baseline=float(v[0]), flash=float(v[1]))
                                             for i, v in enumerate(profiles)])
    (output / 'statistics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for j, metric in enumerate(('spearman', 'kendall')):
        valid = np.isfinite(b[:, j]) & np.isfinite(f[:, j])
        axes[j, 0].hist([b[valid, j], f[valid, j]], bins=np.linspace(-1, 1, 31), label=['Baseline', 'FLASH'])
        axes[j, 0].set(title=f'{metric}: matched context correlations', ylabel='Contexts')
        axes[j, 0].legend()
        axes[j, 1].hist(f[valid, j]-b[valid, j], bins=np.linspace(-2, 2, 41))
        axes[j, 1].axvline(0, color='black', linestyle='--')
        axes[j, 1].set(title=f'{metric}: FLASH minus Baseline', ylabel='Contexts')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'correlation_distributions.{extension}', dpi=160)
    plt.close(fig)
    def fmt(x):
        return '계산 불가' if x is None else f'{x:.4f}'
    def estimate(stats, i):
        if stats['means'] is None:
            return '계산 불가'
        text = fmt(stats['means'][i])
        return text + (f" [{stats['ci95'][i][0]:.4f}, {stats['ci95'][i][1]:.4f}]" if stats['ci95'] else ' [CI 계산 불가]')
    lines = ['# Frostbite 가치 함수 비교', '',
             f"평가 프레임 {len(b):,}개, 에피소드 {result['evaluated_episodes']}개, 블록 {result['evaluated_blocks']}개.",
             f'Bootstrap: {unit} 단위 paired 재표집 {repetitions:,}회, percentile 95% CI, seed={seed}.', '',
             '| 상관계수 | Baseline 평균 [95% CI] | FLASH 평균 [95% CI] | 차이 [95% CI] |',
             '|---|---:|---:|---:|']
    for metric, stats in result['metrics'].items():
        lines.append(f"| {metric} | {estimate(stats, 0)} | {estimate(stats, 1)} | {estimate(stats, 2)} |")
    lines += ['', '평균 차이가 절대 효과 크기입니다. 차이의 CI가 0을 포함하는지로 통계적 불확실성을 확인합니다.',
              '이 구간은 두 고정 체크포인트의 평가 문맥에 대한 불확실성입니다. 학습 시드 간 변동을 포함하지 않습니다.',
              '에피소드 수도 작다면 신뢰구간을 탐색적으로 해석하세요. 유효 재표집 단위가 1개뿐이면 CI를 계산하지 않습니다.',
              'block을 선택하면 블록 사이의 의존성은 반영하지 못합니다. 기본값 episode를 권장합니다.', '',
              '| 지표 | Baseline 중앙값 [Q1, Q3] | FLASH 중앙값 [Q1, Q3] | 차이 중앙값 [Q1, Q3] | FLASH 개선 비율 | 제외 쌍 |',
              '|---|---:|---:|---:|---:|---:|']
    for metric, stats in result['metrics'].items():
        cells = []
        for name in ('baseline', 'flash', 'difference'):
            d = stats[name]
            cells.append(f"{d['median']:.4f} [{d['q1']:.4f}, {d['q3']:.4f}]" if d else '계산 불가')
        fraction = f"{100*stats['improved_fraction']:.1f}%" if stats['improved_fraction'] is not None else '계산 불가'
        lines.append(f"| {metric} | {' | '.join(cells)} | {fraction} | {stats['excluded_pairs']} |")
    lines += ['', '한 평가 문맥은 동일한 관측·행동 이력에 이글루 17단계 변형을 적용한 한 평가 시점입니다.',
              '예측이 상수인 경우 상관계수는 정의되지 않습니다. 해당 비교 쌍을 제외하며 0으로 대체하지 않습니다.',
              '개별 값과 정의 불가 개수는 per_context_correlations.csv 및 statistics.json에 저장합니다.',
              '평균 개선이 관찰되더라도 절대 상관계수의 크기를 함께 해석해야 하며, 이 결과만으로 공간적 얽힘의 원인이나 해소를 입증하지 않습니다.',
              '', '![상관계수 분포](correlation_distributions.png)', '']
    report = '\n'.join(lines)
    (output / 'report_ko.md').write_text(report, encoding='utf-8')
    print(report.split('![상관계수 분포]')[0])
    print(f'결과 저장: {output.resolve()}')
    return result
