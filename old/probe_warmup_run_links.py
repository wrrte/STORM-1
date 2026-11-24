"""Prove warmup-to-run linkage from saved W&B arguments and evaluation summaries.

This offline probe does not claim the W&B server was queried successfully.
No credentials are read by this script. It exports only selected run fields.
"""
import csv
import json
from pathlib import Path
import statistics
import sys

import yaml
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from queue_train import canonical_game
from warmup_registry import DEFAULT_CACHE, aggregate

PROJECT = 'choemj-kaist/STORM'
SAMPLES = [
    ('Boxing', 6040, 'A6000', 'ckpt/Boxing-6040_Shared/shared_warmup_50048'),
    ('Kangaroo', 6030, 'A6000', 'ckpt/Kangaroo-6030_Shared/shared_warmup_50067'),
    ('Freeway', 6020, 'A6000', 'ckpt/Freeway-6020_Shared/shared_warmup_51200'),
]
OUTPUT = ROOT / 'results/warmup_wandb_probe_20261123'


def argument(args, flag):
    if flag not in args or args.index(flag) + 1 == len(args):
        return None
    return args[args.index(flag) + 1]


def checkpoint_name(value):
    parts = Path(value).parts
    if 'ckpt' not in parts:
        return None
    return str(Path(*parts[parts.index('ckpt'):]))


def extract():
    registry = aggregate(DEFAULT_CACHE, 'warmups-20261123')
    requested = {source: (game, seed, model) for game, seed, model, source in SAMPLES}
    for source, (game, seed, model) in requested.items():
        assert any(item['source'] == source for item in registry['games'][game][model][str(seed)]['snapshots'])
    classification = ROOT / 'results/wandb_runs_classification.csv'
    with classification.open(newline='') as handle:
        catalog = {row['Run ID']: row for row in csv.DictReader(handle)}
    records = []
    for metadata_path in sorted((ROOT/'wandb').glob('*/files/wandb-metadata.json')):
        metadata = json.loads(metadata_path.read_text())
        args = metadata.get('args', [])
        resume = argument(args, '--resume_from') or argument(args, '--resume_warmup')
        if not resume or checkpoint_name(resume) not in requested:
            continue
        source = checkpoint_name(resume)
        game, seed, model = requested[source]
        # A checkpoint path plus explicit game and seed identifies this continuation.
        assert int(argument(args, '-seed')) == seed
        assert canonical_game(argument(args, '-env_name')) == game
        run_id = metadata_path.parent.parent.name.rsplit('-', 1)[-1]
        summary_path = metadata_path.parent/'wandb-summary.json'
        config_path = metadata_path.parent/'config.yaml'
        if not summary_path.exists() or not config_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        raw = yaml.safe_load(config_path.read_text())
        config = {key: value.get('value') if isinstance(value, dict) and 'value' in value else value
                  for key, value in raw.items()}
        training = config.get('JointTrainAgent', {})
        retrieval = training.get('Retrieval', {})
        episodes = {int(key.rsplit('_', 1)[-1]): value for key, value in summary.items()
                    if key.startswith('eval/episode_return_') and key.rsplit('_', 1)[-1].isdigit()}
        reported = summary.get('eval/episode_avg_return')
        computed = statistics.mean(episodes.values()) if episodes else None
        mean_verified = reported is not None and computed is not None and abs(reported - computed) < 1e-8
        prior = catalog.get(run_id, {})
        csv_value = prior.get('Eval Return')
        csv_verified = bool(csv_value and csv_value != 'N/A' and reported is not None and
                            abs(float(csv_value) - reported) < 1e-8)
        records.append(dict(game=game, seed=seed, gpu_type=model, warmup_source=source,
            run_id=run_id, run_name=prior.get('Run Name'),
            branch=argument(args, '--branch_experiment') or argument(args, '--branch_mode') or 'unknown',
            url=f'https://wandb.ai/{PROJECT}/runs/{run_id}',
            metric='eval/episode_avg_return', value=reported, episode_count=len(episodes),
            recomputed_mean=computed, mean_verified=mean_verified, classification_verified=csv_verified,
            cached_state=prior.get('State'), training_end=training.get('SampleMaxSteps'),
            retrieval_enable=retrieval.get('enable'), hash_bits=retrieval.get('hash_bits'),
            target=retrieval.get('target'), anchor_offset=retrieval.get('anchor_offset'),
            value_signal=retrieval.get('value_signal'), score_combination=retrieval.get('score_combination'),
            demonstration=training.get('UseDemonstration'),
            evaluation_episodes=[dict(index=index, value=value) for index, value in sorted(episodes.items())],
            evidence=dict(match='exact_checkpoint_path_and_game_and_seed', resume_argument=resume,
                          metadata=str(metadata_path.relative_to(ROOT)), summary=str(summary_path.relative_to(ROOT)),
                          config=str(config_path.relative_to(ROOT))),
            data_source='local_wandb_cache', online_verified=False))
    return sorted(records, key=lambda row: (row['game'], row['seed'], row['branch'], row['run_id']))


def workbook(records):
    book = Workbook()
    book.remove(book.active)

    def sheet(name, headers, rows, widths, table_name):
        ws = book.create_sheet(name)
        ws.sheet_view.showGridLines = False
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
        ws.cell(1, 1, '로컬 W&B 로그 샘플 · 온라인 API 조회는 인증 오류로 미완료')
        ws.cell(1, 1).font = Font(name='맑은 고딕', bold=True, size=13, color='9A6700')
        ws.cell(1, 1).alignment = Alignment(wrap_text=True, vertical='center')
        ws.row_dimensions[1].height = 34
        ws.append(headers)
        for cell in ws[2]:
            cell.fill = PatternFill('solid',fgColor='18334D')
            cell.font = Font(name='맑은 고딕',bold=True,color='FFFFFF')
        for values in rows:
            ws.append(values)
        for row in ws.iter_rows(min_row=3):
            for cell in row:
                cell.alignment = Alignment(vertical='top',wrap_text=True)
                cell.font = Font(name='맑은 고딕',size=11)
            ws.row_dimensions[row[0].row].height = 34
        for col,width in enumerate(widths,1):
            ws.column_dimensions[get_column_letter(col)].width = width
        table = Table(displayName=table_name,ref=f'A2:{get_column_letter(len(headers))}{ws.max_row}')
        table.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2',showRowStripes=True)
        ws.add_table(table)
        ws.freeze_panes='D3'
        return ws

    grouped = []
    for game,seed,model,source in SAMPLES:
        matched = [row for row in records if row['warmup_source']==source]
        values = {row['branch']:row['value'] for row in matched}
        grouped.append([game,f'{model}:{seed}',source,len(matched),
                        values.get('baseline'),values.get('retrieval'),values.get('target1'),
                        values.get('value'),values.get('add')])
    sheet('warmup별 결과',['게임','GPU:시드','shared warmup','연결 run 수','baseline','retrieval','target1','value','add'],
          grouped,[20,20,70,15,14,14,14,14,14],'WarmupSampleSummary')
    run_rows = [[r['game'],f"{r['gpu_type']}:{r['seed']}",r['branch'],r['run_id'],r['value'],r['episode_count'],
                 r['mean_verified'],r['cached_state'],r['hash_bits'],r['target'],r['anchor_offset'],
                 r['value_signal'],r['score_combination'],r['demonstration'],r['warmup_source'],r['url']]
                for r in records]
    ws = sheet('run별 상세',['게임','GPU:시드','분기','Run ID','eval/episode_avg_return','평가 에피소드 수',
               '평균 재계산 일치','저장된 run 상태','hash_bits','target','anchor_offset','value_signal',
               'score_combination','demonstration','shared warmup','W&B 링크'],run_rows,
               [20,20,16,15,28,20,20,20,14,14,18,18,23,20,70,65],'ContinuationRunDetails')
    for row in ws.iter_rows(min_row=3):
        row[-1].hyperlink = row[-1].value
        row[-1].style = 'Hyperlink'
    episode_rows = [[r['game'],f"{r['gpu_type']}:{r['seed']}",r['branch'],r['run_id'],e['index'],e['value']]
                    for r in records for e in r['evaluation_episodes']]
    sheet('평가 에피소드',['게임','GPU:시드','분기','Run ID','에피소드 번호','평가 return'],episode_rows,
          [20,20,16,15,20,20],'EvaluationEpisodes')
    evidence_rows = [[r['run_id'],r['warmup_source'],r['evidence']['resume_argument'],r['evidence']['metadata'],
                      r['evidence']['summary'],r['classification_verified'],'local_wandb_cache'] for r in records]
    sheet('연결 근거',['Run ID','shared warmup','실행 인자 resume 경로','W&B metadata 파일','W&B summary 파일',
                       '기존 W&B CSV와 일치','데이터 출처'],evidence_rows,[15,70,85,80,80,25,25],'LinkEvidence')
    return book


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    records = extract()
    assert len(records) > 0
    assert all(row['mean_verified'] and row['episode_count']==20 for row in records)
    payload = dict(api_access='authentication_failed', public_access='no_run_returned',
                   source='local_wandb_cache', online_verified=False,
                   samples=[dict(game=g,seed=s,gpu_type=m,source=p) for g,s,m,p in SAMPLES],runs=records)
    (OUTPUT/'sample_results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    xlsx = OUTPUT/'warmup_run_sample.xlsx'
    workbook(records).save(xlsx)
    check=load_workbook(xlsx,data_only=True)
    assert check['run별 상세'].max_row-2==len(records)
    assert check['평가 에피소드'].max_row-2==20*len(records)
    for game,seed,model,source in SAMPLES:
        matched=[row for row in records if row['warmup_source']==source]
        print(game,seed,'runs',len(matched),{row['branch']:row['value'] for row in matched})
    print('All means verified from 20 episode returns:',all(row['mean_verified'] for row in records))
    print('Matched existing W&B export:',sum(row['classification_verified'] for row in records),'/',len(records))
    print('XLSX',xlsx)
    print('JSON',OUTPUT/'sample_results.json')


if __name__=='__main__':
    main()
