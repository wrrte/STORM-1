"""Export matched interim and full comparisons for the current hash/target campaign.

    python results/convert_csv_to_excel.py

Refresh collection first with classify_wandb_runs.py --excel or --offline --excel.
Historical CSV files, manual scores, and previous experiment backups are not used.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
from math import fsum
import os
from pathlib import Path
from statistics import fmean, median, stdev
import tempfile
from zoneinfo import ZoneInfo

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from hash_target_common import (
    expected_rows, load_manifest, number, read_csv_rows, report_paths,
)

# Keep these literal constants readable by baseline_campaign.py. The report uses
# only Random/Human normalization constants, never paper/reproduced run scores.
REFERENCE_SCORES = {
    'Alien': (227.8, 7127.7, 984), 'Amidar': (5.8, 1719.5, 205),
    'Assault': (222.4, 742.0, 801), 'Asterix': (210.0, 8503.3, 1028),
    'BankHeist': (14.2, 753.1, 641), 'BattleZone': (2360.0, 37187.5, 13540),
    'Boxing': (0.1, 12.1, 80), 'Breakout': (1.7, 30.5, 16),
    'ChopperCommand': (811.0, 7387.8, 1888), 'CrazyClimber': (10780.5, 35829.4, 66776),
    'DemonAttack': (152.1, 1971.0, 165), 'Freeway': (0.0, 29.6, 34),
    'Frostbite': (65.2, 4334.7, 1316), 'Gopher': (257.6, 2412.5, 8240),
    'Hero': (1027.0, 30826.4, 11044), 'Jamesbond': (29.0, 302.8, 509),
    'Kangaroo': (52.0, 3035.0, 4208), 'Krull': (1598.0, 2665.5, 8413),
    'KungFuMaster': (258.5, 22736.3, 26182), 'MsPacman': (307.3, 6951.6, 2673),
    'Pong': (-20.7, 14.6, 11), 'PrivateEye': (24.9, 69571.3, 7781),
    'Qbert': (163.9, 13455.0, 4522), 'RoadRunner': (11.5, 7845.0, 17564),
    'Seaquest': (68.4, 42054.7, 525), 'UpNDown': (533.4, 11693.2, 7985),
}
REPRODUCED_SCORES = {
    'Alien': 748, 'Amidar': 144, 'Assault': 1377, 'Asterix': 1318,
    'BankHeist': 990, 'BattleZone': 5830, 'Boxing': 81, 'Breakout': 41,
    'ChopperCommand': 1644, 'CrazyClimber': 79196, 'DemonAttack': 324, 'Freeway': 0,
    'Frostbite': 366, 'Gopher': 5307, 'Hero': 11434, 'Jamesbond': 408,
    'Kangaroo': 3512, 'Krull': 6522, 'KungFuMaster': 20046, 'MsPacman': 1490,
    'Pong': 18, 'PrivateEye': 100, 'Qbert': 2910, 'RoadRunner': 14841,
    'Seaquest': 557, 'UpNDown': 6128,
}

NAVY, LIGHT, GREEN, GRAY = '172B4D', 'F5F8FC', 'DFF2E7', 'E6EDF5'
FONT = '맑은 고딕'
STATE_LABELS = {'completed': '완료', 'running': '진행 중', 'pending': '대기',
                'failed': '실패', 'interrupted': '중단', 'invalid': '검증 오류', 'evaluating': '평가 종료 대기'}
IQM_DESCRIPTION = ('각 게임에 같은 총 가중치를 주고 공통 시드에 균등하게 나눕니다. '
                   'HNS를 정렬해 누적 가중치 25~75% 구간을 평균하며, 경계 점수는 해당 구간에 걸친 비중만 반영합니다.')


def hns(game, score):
    random, human, _ = REFERENCE_SCORES[game]
    return (score - random) / (human - random)


def game_weighted_iqm(by_game):
    weighted_scores = []
    for scores in by_game.values():
        weight = Fraction(1, len(by_game) * len(scores))
        weighted_scores.extend((score, weight) for score in scores)

    lower, upper = Fraction(1, 4), Fraction(3, 4)
    cumulative = Fraction(0)
    contributions = []
    for score, weight in sorted(weighted_scores):
        end = cumulative + weight
        retained = min(end, upper) - max(cumulative, lower)
        if retained > 0:
            contributions.append(score * float(retained / (upper - lower)))
        if end >= upper:
            break
        cumulative = end
    return fsum(contributions)


def calculate_metrics(by_game):
    """Compute game-weighted aggregate metrics on the matched sample."""
    if not by_game:
        return dict(mean=None, median=None, iqm=None, gap=None)
    game_means = [fmean(values) for values in by_game.values()]
    return dict(mean=fmean(game_means), median=median(game_means),
                iqm=game_weighted_iqm(by_game),
                gap=fmean(max(0, 1 - value) for value in game_means))


def select_cohorts(completed, job_ids):
    """Return the settings-first and samples-first matched comparisons."""
    order = {setting: index for index, setting in enumerate(completed)}
    per_job = {tuple(setting for setting, jobs in completed.items() if identity in jobs)
               for identity in job_ids}
    max_settings = max(map(len, per_job), default=0)
    if not max_settings:
        empty = frozenset(), ()
        return empty, empty

    # Any set of comparable settings must occur together on at least one job.
    overall = min(
        ((frozenset.intersection(*(completed[setting] for setting in settings)), settings)
         for settings in per_job if len(settings) == max_settings),
        key=lambda choice: (-len(choice[0]), tuple(order[setting] for setting in choice[1])),
    )
    if max_settings == 1:
        return overall, overall

    # A sample shared by at least two settings is contained in their intersection.
    intersections = {left & right for left, right in combinations(completed.values(), 2)}
    max_samples = max(map(len, intersections))
    interim = min(
        ((jobs, tuple(setting for setting, done in completed.items() if jobs <= done))
         for jobs in intersections if len(jobs) == max_samples),
        key=lambda choice: (-len(choice[1]), tuple(order[setting] for setting in choice[1])),
    )
    return overall, interim


def comparison_statistics(jobs, games, settings, records, cohort):
    common, selected = cohort
    common_by_game = {game: [identity for identity in jobs if identity in common and jobs[identity]['game'] == game]
                      for game in games}
    baseline = bool(common) and all(records[identity, 'Baseline']['Eval Return'] is not None for identity in common)
    scored = {*selected, 'Baseline'} if baseline else set(selected)
    statistics, metrics = {}, {}
    for setting in settings:
        by_game = {}
        for game, identities in common_by_game.items():
            scores = [records[identity, setting]['Eval Return'] for identity in identities] if setting in scored else []
            controls = [records[identity, 'Baseline']['Eval Return'] for identity in identities] if scores and baseline else []
            avg = fmean(scores) if scores else None
            control = fmean(controls) if controls else None
            delta = avg - control if controls else None
            statistics[game, setting] = dict(n=len(scores), mean=avg,
                sd=stdev(scores) if len(scores)>1 else None, hns=hns(game, avg) if scores else None,
                control=control, delta=delta,
                delta_hns=delta/(REFERENCE_SCORES[game][1]-REFERENCE_SCORES[game][0]) if controls else None,
                improvement=delta/abs(control) if controls and control != 0 else None,
                wins=sum(score>base for score,base in zip(scores,controls)) if controls else None)
            if scores:
                by_game[game] = [hns(game, score) for score in scores]
        metrics[setting] = calculate_metrics(by_game)
    ordered = sorted(selected, key=lambda setting: -metrics[setting]['mean'])
    ranks = {setting: 1 + sum(metrics[other]['mean'] > metrics[setting]['mean'] for other in ordered)
             for setting in ordered} if len(ordered) > 1 else {}
    return dict(common=common, selected=selected, baseline=baseline, scored=scored,
                common_by_game=common_by_game, stats=statistics, metrics=metrics, ranked=ordered, ranks=ranks)


def aggregate(manifest, csv_rows):
    jobs = {job['id']: job for job in manifest['jobs']}
    games = sorted({job['game'] for job in jobs.values()})
    if set(games) - REFERENCE_SCORES.keys():
        raise ValueError(f'Missing Random/Human references: {set(games)-REFERENCE_SCORES.keys()}')
    settings = ['Baseline'] + [f'h{h} / t{t}' for h, t in manifest['pairs']]
    rows = {}
    for _, _, expected in expected_rows(manifest):
        key = expected['Job ID'], expected['Stage']
        row = dict(expected, State='pending', **{'Eval Return': None, 'Notes': ''})
        row.update(csv_rows.get(key, {}))
        if row['State'] not in STATE_LABELS:
            raise ValueError(f'Unknown result state: {row["State"]}')
        row['Eval Return'] = number(row.get('Eval Return')) if row['State'] == 'completed' else None
        rows[key] = row
    records = {(row['Job ID'], row['Setting']): row for row in rows.values()}
    # A job is one frozen (game, GPU model, seed, shared warmup) observation.
    completed = {setting: frozenset(identity for identity in jobs if records[identity, setting]['Eval Return'] is not None)
                 for setting in settings}
    overall, interim = select_cohorts({setting: completed[setting] for setting in settings[1:]}, jobs)
    comparison = comparison_statistics(jobs, games, settings, records, overall)
    return dict(jobs=jobs, games=games, settings=settings, rows=rows, records=records,
                counts={setting: len(done) for setting, done in completed.items()}, **comparison,
                interim=comparison_statistics(jobs, games, settings, records, interim),
                provisional=any(len(done) < len(jobs) for done in completed.values()))


def set_cell(ws, row, col, value):
    cell = ws.cell(row, col, value)
    if isinstance(value, str):
        cell.data_type = 's'
    return cell


def sheet(wb, name, title, subtitle, last_col):
    ws = wb.create_sheet(name)
    ws.sheet_view.showGridLines = False
    ws.sheet_view.zoomScale = 85
    ws.sheet_properties.tabColor = '087F8C'
    for row, text, size, color in [(1,title,18,NAVY),(2,subtitle,10,'61758A')]:
        ws.merge_cells(start_row=row,start_column=1,end_row=row,end_column=last_col)
        cell = set_cell(ws,row,1,text)
        cell.font = Font(name=FONT,size=size,bold=row==1,color=color)
        cell.alignment = Alignment(vertical='center',wrap_text=True)
        ws.row_dimensions[row].height = 34 if row==1 else 30
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = 'landscape'
    ws.page_setup.paperSize = ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth,ws.page_setup.fitToHeight = 1,0
    ws.oddFooter.right.text = 'Page &P / &N'
    return ws


def headers(ws,row,labels):
    for col,label in enumerate(labels,1):
        cell = set_cell(ws,row,col,label)
        cell.fill = PatternFill('solid',fgColor=NAVY)
        cell.font = Font(name=FONT,size=10,bold=True,color='FFFFFF')
        cell.alignment = Alignment(horizontal='center',vertical='center',wrap_text=True)
    ws.row_dimensions[row].height = 34


def line(ws,row,values,control=False,best=False):
    for col,value in enumerate(values,1):
        cell = set_cell(ws,row,col,value)
        cell.font = Font(name=FONT,size=10,color=NAVY,bold=control or best)
        cell.alignment = Alignment(vertical='center',horizontal='left' if isinstance(value,str) else 'right')
        cell.fill = PatternFill('solid',fgColor=GRAY if control else GREEN if best else LIGHT if row%2 else 'FFFFFF')
        cell.border = Border(bottom=Side(style='hair',color='DCE5EF'))
    ws.row_dimensions[row].height = 24


def table(ws,name,row,last,col):
    item = Table(displayName=name,ref=f'A{row}:{get_column_letter(col)}{last}')
    item.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2',showRowStripes=False,showColumnStripes=False)
    ws.add_table(item)


def widths(ws,values):
    for col,width in enumerate(values,1):
        ws.column_dimensions[get_column_letter(col)].width = width


def formats(ws,row,values):
    for col,pattern in values.items():
        ws.cell(row,col).number_format = pattern


def parameters(setting,manifest):
    if setting == 'Baseline':
        return '—','—'
    return next((h,t) for h,t in manifest['pairs'] if setting==f'h{h} / t{t}')


def comparison_status(data, comparison):
    settings = len(comparison['selected'])
    games = sum(bool(jobs) for jobs in comparison['common_by_game'].values())
    if not settings:
        status = '최종 평가 완료 대기'
    elif settings == 1:
        status = '조합 간 순위 대기'
    else:
        status = 'Mean HNS 잠정 순위' if data['provisional'] else '전체 완료'
    return (f'조합 {settings}/{len(data["settings"])-1}개 · '
            f'사용 표본 {len(comparison["common"])}/{len(data["jobs"])}개 · '
            f'{games}/{len(data["games"])}게임 | {status}')


def add_interim_comparison(wb, manifest, data):
    comparison = data['interim']
    settings = (['Baseline'] if comparison['baseline'] else []) + comparison['ranked']
    jobs = sorted((data['jobs'][identity] for identity in comparison['common']),
                  key=lambda job: (job['game'], job['gpu_type'], job['seed'], job['id']))
    games = [game for game in data['games'] if comparison['common_by_game'][game]]
    labels = ['순위', '설정', 'hash_bits', 'target', 'Mean HNS', 'Median HNS', 'IQM', 'Optimality Gap',
              *[f'{game} 평균 점수' for game in games]]
    sample_labels = ['게임', 'GPU 모델', '시드', '실제 warmup step', *settings]
    last_col = max(len(labels), len(sample_labels))
    ws = sheet(wb, '중간 비교', '중간 비교 | 게임·시드 표본 수 우선', comparison_status(data, comparison), last_col)
    headers(ws, 4, labels)
    for row, setting in enumerate(settings, 5):
        metrics = comparison['metrics'][setting]
        means = [comparison['stats'][game, setting]['mean'] for game in games]
        rank = comparison['ranks'].get(setting, '—')
        line(ws, row, [rank, setting, *parameters(setting, manifest), metrics['mean'], metrics['median'], metrics['iqm'],
                       metrics['gap'], *means], control=setting == 'Baseline', best=rank == 1)
        formats(ws, row, {**{col: '0.0000' for col in range(5, 9)},
                          **{col: '#,##0.00' for col in range(9, len(labels) + 1)}})
    if settings:
        table(ws, 'InterimComparison', 4, 4 + len(settings), len(labels))
    ws['E4'].comment = Comment('각 게임의 공통 warmup 평균 HNS를 구한 뒤 게임에 동일한 가중치를 적용합니다.', 'STORM')
    ws['G4'].comment = Comment(IQM_DESCRIPTION, 'STORM')

    top = 7 + len(settings)
    ws.merge_cells(start_row=top, start_column=1, end_row=top, end_column=last_col)
    ws.cell(top, 1, '사용한 게임·시드별 최종 점수').font = Font(name=FONT, bold=True, color=NAVY)
    headers(ws, top + 1, sample_labels)
    for row, job in enumerate(jobs, top + 2):
        scores = [data['records'][job['id'], setting]['Eval Return'] for setting in settings]
        line(ws, row, [job['game'], job['gpu_type'], job['seed'], job['warmup_step'], *scores])
        formats(ws, row, {col: '#,##0.00' for col in range(5, len(sample_labels) + 1)})
        ws.cell(row, 1).comment = Comment(job['warmup_source'], 'STORM')
    if jobs:
        table(ws, 'InterimSeedScores', top + 1, top + 1 + len(jobs), len(sample_labels))
    widths(ws, [19, 18, 11, 19, 16, 14, 14, 19, *([22] * (last_col - 8))])
    ws.freeze_panes = 'E5'


def add_heatmaps(ws,start,manifest,data):
    bits = sorted({h for h,t in manifest['pairs']})
    targets = sorted({t for h,t in manifest['pairs']})
    width = len(targets)+1
    right = max(7,width+3)
    for name,key,top,left,higher in [
        ('Mean HNS','mean',start,1,True),('Median HNS','median',start,right,True),
        ('IQM','iqm',start+len(bits)+6,1,True),('Optimality Gap','gap',start+len(bits)+6,right,False)]:
        ws.merge_cells(start_row=top,start_column=left,end_row=top,end_column=left+width-1)
        cell = set_cell(ws,top,left,f'{name} {"↑" if higher else "↓"}')
        cell.font = Font(name=FONT,size=12,bold=True,color=NAVY)
        cell.fill = PatternFill('solid',fgColor='EEF4FA')
        ws.row_dimensions[top].height = 28
        header = top+2
        for offset,label in enumerate(['h / t',*targets]):
            cell = set_cell(ws,header,left+offset,label)
            cell.font = Font(name=FONT,bold=True,color='FFFFFF',size=10)
            cell.fill = PatternFill('solid',fgColor=NAVY)
            cell.alignment = Alignment(horizontal='center')
        for rr,h in enumerate(bits,header+1):
            cell = set_cell(ws,rr,left,h)
            cell.font = Font(name=FONT,bold=True,color=NAVY)
            cell.fill = PatternFill('solid',fgColor='EEF4FA')
            cell.alignment = Alignment(horizontal='center')
            ws.row_dimensions[rr].height = 29
            for offset,t in enumerate(targets,1):
                value = data['metrics'].get(f'h{h} / t{t}',{}).get(key)
                cell = set_cell(ws,rr,left+offset,value)
                cell.number_format = '0.0000'
                cell.font = Font(name=FONT,size=11,color=NAVY)
                cell.alignment = Alignment(horizontal='center')
                cell.fill = PatternFill('solid',fgColor=GRAY)
        ref = f'{get_column_letter(left+1)}{header+1}:{get_column_letter(left+width-1)}{header+len(bits)}'
        ws.conditional_formatting.add(ref,ColorScaleRule(start_type='min',start_color='FFF3D6' if higher else '75C3AB',
                                      end_type='max',end_color='75C3AB' if higher else 'FFF3D6'))
    return start+2*len(bits)+11


def build_workbook(manifest,data,manifest_path):
    wb = Workbook()
    wb.remove(wb.active)
    wb.properties.title = 'STORM hash_bits × target comparison'
    wb.properties.description = manifest['campaign_id']
    games,settings = data['games'],data['settings']
    last_col = max(10,9+len(games))
    ws = sheet(wb,'전체 비교','전체 비교 | 해시·타깃 조합 수 우선',manifest['campaign_id'],last_col)
    ws.merge_cells(start_row=3,start_column=1,end_row=3,end_column=last_col)
    timestamp = datetime.now(timezone.utc).astimezone(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
    ws['A3'] = f'{comparison_status(data, data)} | {timestamp}'
    ws['A3'].font = Font(name=FONT,size=10,color='9C6500' if data['provisional'] else '087F8C')
    labels = ['순위','설정','hash_bits','target','완료 / 예정','Mean HNS','Median HNS','IQM','Optimality Gap',*[f'{game} 평균 점수' for game in games]]
    headers(ws,4,labels)
    overview_settings = ['Baseline', *data['ranked'], *(s for s in settings[1:] if s not in data['selected'])]
    for row,setting in enumerate(overview_settings,5):
        metric = data['metrics'][setting]
        rank = data['ranks'].get(setting,'—')
        h,t = parameters(setting,manifest)
        line(ws,row,[rank,setting,h,t,f'{data["counts"][setting]} / {len(data["jobs"])}',
                     metric['mean'],metric['median'],metric['iqm'],metric['gap'],
                     *[data['stats'][game,setting]['mean'] for game in games]],control=setting=='Baseline',best=rank==1)
        formats(ws,row,{**{c:'0.0000' for c in range(6,10)},**{c:'#,##0.00' for c in range(10,len(labels)+1)}})
    for address,text in {
        'A4':'같은 게임·시드에서 평가를 마친 조합이 둘 이상이면 Mean HNS 내림차순으로 순위를 매깁니다.',
        'E4':'이 설정의 최종 평가 완료 수 / 예정 warmup 수. 일부 설정만 완료된 warmup도 진행률에는 반영합니다.',
        'F4':'게임별 공통 warmup의 평균 HNS를 게임 간 동일 가중 평균합니다.',
        'G4':'게임별 공통 warmup 평균 HNS의 중앙값입니다.',
        'H4':IQM_DESCRIPTION,
        'I4':'게임별 max(0, 1 − 공통 warmup 평균 HNS)의 평균. 낮을수록 좋습니다.',
    }.items():
        ws[address].comment = Comment(text,'STORM')
    end = 4+len(settings)
    table(ws,'Overview',4,end,len(labels))
    last = add_heatmaps(ws,end+3,manifest,data)
    widths(ws,[8,18,12,11,16,16,16,16,19,*([22]*len(games))])
    ws.freeze_panes = 'F5'
    ws.print_title_rows = '1:4'
    ws.print_area = f'A1:{get_column_letter(last_col)}{last}'

    add_interim_comparison(wb, manifest, data)
    ws = sheet(wb,'게임별 비교','게임별 | 공통 완료 warmup의 설정 비교',
               '전체 비교와 같은 조합·게임·시드 사용 · SD는 시드 점수의 표본 표준편차',12)
    for index,game in enumerate(games):
        top = 5+index*(len(settings)+4)
        ws.merge_cells(start_row=top,start_column=1,end_row=top,end_column=12)
        ws.cell(top,1,f'{game} · 공통 완료 {len(data["common_by_game"][game])}개 warmup').font = Font(name=FONT,bold=True,color=NAVY,size=12)
        headers(ws,top+1,['설정','hash_bits','target','공통 warmup n','평균 점수','시드 간 SD','평균 HNS','Baseline 공통 평균','Δ Score','Δ HNS','개선율','승리 시드'])
        available = [s for s in settings[1:] if data['stats'][game,s]['mean'] is not None]
        best = max(available,key=lambda s:data['stats'][game,s]['mean']) if len(available) > 1 else None
        for offset,setting in enumerate(settings,top+2):
            st = data['stats'][game,setting]
            h,t = parameters(setting,manifest)
            line(ws,offset,[setting,h,t,st['n'],st['mean'],st['sd'],st['hns'],st['control'],st['delta'],st['delta_hns'],st['improvement'],
                            f'{st["wins"]} / {st["n"]}' if st['wins'] is not None and setting!='Baseline' else None],control=setting=='Baseline',best=setting==best)
            formats(ws,offset,{5:'#,##0.00',6:'#,##0.00',7:'0.0000',8:'#,##0.00',9:'+#,##0.00;-#,##0.00;0.00',10:'+0.0000;-0.0000;0.0000',11:'+0.0%;-0.0%;0.0%'})
        table(ws,f'Game{index}',top+1,top+1+len(settings),12)
    widths(ws,[19,12,11,18,18,16,16,22,18,16,14,14])
    ws.freeze_panes='E7'

    columns = sorted({(job['gpu_type'],job['seed']) for job in data['jobs'].values()})
    models = sorted({gpu for gpu,seed in columns})
    ws = sheet(wb,'GPU·시드별 점수','GPU·시드별 | 부분 완료 결과 포함',
               '개별 완료 점수를 즉시 표시 · 완료분 평균은 진행 상황 확인용이며 전체 비교의 공통 표본 평균과 다를 수 있습니다.',6+len(columns)+len(models))
    for gpu in models:
        positions = [7+i for i,(model,seed) in enumerate(columns) if model==gpu]
        if len(positions)>1:
            ws.merge_cells(start_row=6,start_column=min(positions),end_row=6,end_column=max(positions))
        ws.cell(6,min(positions),gpu).font=Font(name=FONT,bold=True,color=NAVY)
    headers(ws,7,['게임','설정','hash_bits','target','완료분 평균 [참고]','완료분 SD',
                  *[f'{gpu}: seed {seed}' for gpu,seed in columns],*[f'{gpu} 평균 [참고]' for gpu in models]])
    unit_lookup = {(job['game'],job['gpu_type'],job['seed']):job['id'] for job in data['jobs'].values()}
    for row,(game,setting) in enumerate(((g,s) for g in games for s in settings),8):
        individual=[]
        for gpu,seed in columns:
            identity=unit_lookup.get((game,gpu,seed))
            individual.append(data['records'].get((identity,setting)))
        scores=[r['Eval Return'] for r in individual if r and r['Eval Return'] is not None]
        model_averages=[]
        for gpu in models:
            selected=[r['Eval Return'] for (model,_),r in zip(columns,individual) if model==gpu and r and r['Eval Return'] is not None]
            model_averages.append(fmean(selected) if selected else None)
        line(ws,row,[game,setting,*parameters(setting,manifest),fmean(scores) if scores else None,stdev(scores) if len(scores)>1 else None,
                     *[None if r is None else r['Eval Return'] if r['Eval Return'] is not None else STATE_LABELS[r['State']] for r in individual],
                     *model_averages],control=setting=='Baseline')
        formats(ws,row,{c:'#,##0.00' for c in range(5,7+len(columns)+len(models))})
        for col,record in enumerate(individual,7):
            if record:
                included = record['Job ID'] in data['common'] and setting in data['scored']
                ws.cell(row,col).comment=Comment(f'Warmup: {record["Warmup ID"]}\nJob: {record["Job ID"]}\n공통 비교 포함: {included}\n{record.get("Notes", "")}','STORM')
                if record['State']!='completed':
                    ws.cell(row,col).fill=PatternFill('solid',fgColor='FFF2CC' if record['State'] in ('running','evaluating') else GRAY)
    table(ws,'SeedScores',7,7+len(games)*len(settings),6+len(columns)+len(models))
    widths(ws,[17,19,12,11,23,17,*([24]*(len(columns)+len(models)))])
    ws.freeze_panes='G8'

    ws=sheet(wb,'실행 상세','실행 상세 | 이번 실험에 배정된 분기만 표시',
             '1행 = 1개 warmup의 1개 설정 · 미완료 점수는 빈칸 · 같은 분기의 재시도는 한 결과만 채택',16)
    headers(ws,6,['게임','GPU 모델','시드','설정','hash_bits','target','상태','평가 평균 점수','HNS','평가 체크포인트','평가 정밀도','평가 episodes',
                  '실제 warmup step','Warmup ID','공통 비교 포함','비고','Run ID','큐 Job ID','Stage','로컬 Run 이름','Warmup 경로','출처','갱신 시각'])
    for rr,record in enumerate(data['rows'].values(),7):
        score=record['Eval Return']
        included = record['Job ID'] in data['common'] and record['Setting'] in data['scored']
        line(ws,rr,[record['Game'],record['GPU Type'],int(record['Seed']),record['Setting'],record['Hash Bits'],record['Retrieval Target'],
                    STATE_LABELS[record['State']],score,hns(record['Game'],score) if score is not None else None,
                    number(record.get('Eval Checkpoint Step')),record.get('Eval Precision'),number(record.get('Eval Episodes')),
                    int(record['Actual Warmup Step']),record['Warmup ID'],included,record.get('Notes'),
                    record.get('Run ID'),record['Job ID'],record['Stage'],record['Local Run Name'],record['Warmup Directory'],record.get('Source'),record.get('Updated At')],
             control=record['Setting']=='Baseline')
        formats(ws,rr,{8:'#,##0.00',9:'0.0000'})
    table(ws,'RunDetails',6,6+len(data['rows']),23)
    widths(ws,[17,16,12,19,12,11,17,20,16,21,17,17,22,26,19,65,20,38,24,65,85,20,30])
    ws.column_dimensions.group('Q','W',outline_level=1,hidden=True)
    ws.freeze_panes='H7'

    ws=sheet(wb,'집계 기준','집계 기준','범위와 계산 방법',8)
    rules=[
        ('실험 ID',manifest['campaign_id']),('고정 목록',str(Path(manifest_path).resolve())),
        ('대상',f'{len(data["jobs"])}개 warmup · {len(settings)-1}개 hash/target 쌍 + baseline · {len(data["rows"])}개 예정 분기'),
        ('수집 범위','고정 목록의 Job ID와 정확한 로컬 Run 이름에 해당하는 결과만 사용. 과거 CSV·수동 점수·고정 백업은 제외.'),
        ('완료 판정','설정/게임/시드/warmup 및 최종 평가 조건을 검증. 로컬 완료 마커 또는 finished W&B run과 모든 episode 결과가 필요.'),
        ('전체 비교','같은 warmup에서 비교할 수 있는 조합 수를 최대화하고, 동률이면 공통 게임·시드 수가 많은 후보를 선택합니다. 게임별 비교와 히트맵에도 같은 표본을 사용합니다.'),
        ('중간 비교','두 조합 이상에 공통인 게임·시드 수를 최대화하고, 동률이면 조합 수가 많은 후보를 선택합니다. 사용한 게임·시드의 최종 점수를 아래에 표시합니다.'),
        ('Baseline','각 표의 공통 게임·시드에서 모두 완료됐을 때 비교 점수를 표시합니다. Baseline은 조합 수와 순위에서 제외합니다.'),
        ('진행률 / 빈칸','완료/예정에는 표본에 포함되지 않은 분기의 완료도 반영합니다. 시드별 상세는 개별 완료 점수를 즉시 표시합니다.'),
        ('잠정 순위','조합 하나만 집계할 수 있을 때는 점수만 표시합니다. 공통 표본에서 비교할 수 있는 조합이 두 개 이상이면 Mean HNS 내림차순으로 순위를 계산합니다.'),
        ('평가 조건',str(manifest['evaluation'])),
        ('HNS','(score − Random) / (Human − Random). 음수와 1 초과 값을 자르지 않음.'),
        ('Mean / Median','게임별 공통 warmup 평균 HNS의 평균 / 중앙값. 게임별 동일 가중.'),
        ('IQM',IQM_DESCRIPTION),
        ('Optimality Gap','게임별 max(0, 1 − 평균 HNS)의 평균. 낮을수록 좋음.'),
        ('재시도','한 분기를 한 관측으로 집계. 검증된 로컬 결과 우선, 그다음 최신 완료 W&B run. 재시도 평균을 만들지 않음.'),
        ('색상','Baseline 회색. 최고 평균 설정 녹색 행. 히트맵은 모두 진한 녹색이 좋은 성능. 공통 표본 없으면 회색.'),
    ]
    for rr,(label,text) in enumerate(rules,5):
        ws.cell(rr,1,label).font=Font(name=FONT,bold=True,color=NAVY)
        ws.merge_cells(start_row=rr,start_column=2,end_row=rr,end_column=8)
        cell=set_cell(ws,rr,2,text)
        cell.font=Font(name=FONT,size=10,color=NAVY)
        cell.alignment=Alignment(wrap_text=True,vertical='center')
        ws.row_dimensions[rr].height=40
    widths(ws,[26,*([20]*7)])
    wb.active=0
    return wb


def main(csv_path=None,output_path=None,*,manifest_path=None,experiment=None):
    manifest_path,csv_path,output_path=report_paths(experiment,manifest=manifest_path,csv=csv_path,excel=output_path)
    manifest=load_manifest(manifest_path)
    if experiment and manifest.get('experiment_name') not in (None,experiment):
        raise ValueError('The selected manifest belongs to another --experiment')
    rows=read_csv_rows(csv_path,manifest)
    data=aggregate(manifest,rows)
    wb=build_workbook(manifest,data,manifest_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(dir=output_path.parent,prefix='.'+output_path.stem,suffix='.xlsx')
    os.close(descriptor)
    try:
        wb.save(temporary)
        os.replace(temporary,output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f'Excel: {output_path}')
    print(f'  전체 비교  {comparison_status(data, data)}')
    print(f'  중간 비교  {comparison_status(data, data["interim"])}')
    return output_path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment',help='Select the experiment report directory by name')
    parser.add_argument('--csv',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args()
    try:
        main(args.csv,args.output,manifest_path=args.manifest,experiment=args.experiment)
    except (ValueError,OSError) as error:
        parser.exit(1,f'{error}\n')
