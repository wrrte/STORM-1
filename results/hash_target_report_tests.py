"""Regression tests for cohort isolation, evaluation validation and paired reporting.

python -m unittest discover -s results -p hash_target_report_tests.py -v
"""
from copy import deepcopy
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import sys
from types import SimpleNamespace
import unittest

from openpyxl import load_workbook
import yaml

import classify_wandb_runs as collector
import convert_csv_to_excel as report
from hash_target_common import (
    ROOT, FIELDS, PREFIX, atomic_text, build_manifest, config_mismatch, effective_config,
    expected_rows, load_manifest, read_csv_rows, stage_config, report_paths, list_experiments, read_jobs,
)
sys.path.insert(0,str(ROOT))
from queue_train import TrainingQueue

GRID = [[h,t] for h in [9,10,11,12] for t in [4,8,16]]


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.database = self.root/'training_queue/queue.sqlite3'
        self.database.parent.mkdir()
        with sqlite3.connect(self.database) as db:
            db.execute('CREATE TABLE jobs(seq INTEGER PRIMARY KEY,id TEXT,status TEXT,payload TEXT,created REAL,worker TEXT)')
        self.jobs = {}
        for index,(game,gpu,seed) in enumerate([
            ('Alien','3090',3333), ('Alien','3090',3334),
            ('Frostbite','3090',3333), ('Alien','A6000',3333),
        ]):
            self.add_job(f'j{index}',game,gpu,seed)
        self.manifest = build_manifest(self.jobs,self.root,gpu_type=None)
        self.manifest_path = self.root/'campaign.json'
        atomic_text(self.manifest_path,json.dumps(self.manifest))

    def add_job(self,identity,game,gpu,seed):
        snapshot=self.root/'ckpt'/f'{identity}-shared'/'shared_warmup_50100'
        snapshot.mkdir(parents=True)
        (snapshot/'warmup_metadata.json').write_text(json.dumps(dict(seed=seed,env_name=f'ALE/{game}-v5',step=50100)))
        for name in ['training_state.pt','replay_buffer.npz']:(snapshot/name).write_bytes(b'test')
        job=dict(id=identity,name=f'{game}-seed{seed}-queue-{identity}',game=game,gpu_type=gpu,seed=seed,
                 created=1700000000,config_text=(ROOT/'config_files/STORM.yaml').read_text(),
                 overrides={PREFIX+'hash_target_pairs':GRID,PREFIX+'warmup_offset_pairs':[],
                            PREFIX+'enable':'Both',PREFIX+'warmup_steps':50000,PREFIX+'anchor_offset':-2},
                 warmup_source=str(snapshot),resume_warmup=str(snapshot),status='pending',completed_branches=[])
        self.jobs[identity]=job
        with sqlite3.connect(self.database) as db:
            db.execute('INSERT INTO jobs(id,status,payload,created) VALUES(?,?,?,?)',
                       (identity,'pending',json.dumps(job),job['created']))
        return job

    def job_stage(self,identity='j0',key='retrieval_h9_t4'):
        job=next(j for j in self.manifest['jobs'] if j['id']==identity)
        return job,next(s for s in job['stages'] if s['key']==key)

    def write_local(self,identity='j0',key='retrieval_h9_t4',score=1000,marker=True):
        job,stage=self.job_stage(identity,key)
        directory=self.root/'runs'/stage['local_name']
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'config.yaml').write_text(yaml.safe_dump(stage_config(job,stage)))
        data=dict(run_name=stage['local_name'],seed=job['seed'],env_name=f'ALE/{job["game"]}-v5',
                  sample_max_steps=102000,checkpoint_step=100000,precision='bf16',
                  episode_avg_return=score,episode_returns=[score]*20)
        (directory/'final_evaluation.json').write_text(json.dumps(data))
        (directory/'shared_warmup.json').write_text(json.dumps(dict(checkpoint_dir=job['resume_warmup'])))
        if marker:
            location=self.root/'ckpt'/stage['local_name']
            location.mkdir(parents=True,exist_ok=True)
            (location/'queue_stage_complete.json').write_text(json.dumps(dict(stage=f'{identity}:{key}')))
        return job,stage,directory

    def completed_rows(self,identities):
        rows={}
        for job,stage,expected in expected_rows(self.manifest):
            row=dict(expected,State='pending')
            if job['id'] in identities:
                score=1000 if job['game']=='Alien' else 1500
                if stage['key']!='baseline':score+=100
                row.update(collector.evaluation_fields(score,[score]*20,'bf16',100000,20,0,self.manifest))
            rows[job['id'],stage['key']]=row
        return rows

    def fake_run(self,identity='j0',key='retrieval_h9_t4',run_id='new',score=1000,state='finished'):
        job,stage=self.job_stage(identity,key)
        config=stage_config(job,stage)
        config['RunMetadata']=dict(local_run_name=stage['local_name'],game=job['game'],seed=job['seed'])
        args=['--resume_from',job['resume_warmup']]
        def download(root,replace):
            path=Path(root)/'wandb-metadata.json'
            path.write_text(json.dumps(dict(args=args)))
            return path.open()
        summary={'eval/checkpoint_step':100000,'eval/seed':0,'eval/bf16/episode_avg_return':score}
        summary.update({f'eval/bf16/episode_return_{i}':score for i in range(20)})
        return SimpleNamespace(id=run_id,state=state,config=config,summary=summary,
                               created_at='2026-11-24T07:00:00Z',updated_at='2026-11-24T08:00:00Z',
                               file=lambda name:SimpleNamespace(download=download))

    def test_manifest_is_frozen_and_reporting_does_not_mutate_queue(self):
        output=self.root/'runs.csv'
        # Use the already frozen multi-GPU manifest; future matching jobs stay outside.
        self.add_job('future','Alien','3090',5555)
        before=self.database.read_bytes()
        collector.main(output_path=output,offline=True,manifest_path=self.manifest_path,root=self.root)
        after=self.database.read_bytes()
        self.assertEqual(before,after)
        rows=read_csv_rows(output,self.manifest)
        self.assertEqual(len(rows),4*13)
        self.assertNotIn('future',{key[0] for key in rows})
        self.assertEqual(load_manifest(self.manifest_path)['campaign_id'],self.manifest['campaign_id'])

    def test_manifest_tampering_is_detected(self):
        bad=deepcopy(self.manifest)
        bad['jobs'][0]['seed']=999
        self.manifest_path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError,'modified'):load_manifest(self.manifest_path)

    def test_configuration_validation_supports_nested_and_flat_wandb_config(self):
        expected={'Models':{'Agent':{'HiddenDim':512}},'JointTrainAgent':{'EvalSeed':0}}
        flat={'Models.Agent.HiddenDim':512,'JointTrainAgent.EvalSeed':0}
        self.assertIsNone(config_mismatch(flat,expected))
        flat['Models.Agent.HiddenDim']=256
        self.assertIn('HiddenDim',config_mismatch(flat,expected))

    def test_local_final_score_requires_correct_completion_marker(self):
        job,stage,directory=self.write_local(marker=False)
        row=collector.local_result(self.root,job,stage,self.jobs[job['id']],self.manifest)
        self.assertEqual(row['State'],'evaluating')
        self.assertNotIn('Eval Return',row)
        self.write_local()
        row=collector.local_result(self.root,job,stage,self.jobs[job['id']],self.manifest)
        self.assertEqual(row['State'],'completed')
        self.assertEqual(row['Eval Return'],1000)
        marker=self.root/'ckpt'/stage['local_name']/'queue_stage_complete.json'
        marker.write_text(json.dumps(dict(stage='another-job:retrieval_h9_t4')))
        with self.assertRaisesRegex(ValueError,'another'):collector.local_result(self.root,job,stage,self.jobs[job['id']],self.manifest)

    def test_local_rejects_wrong_eval_step_nan_config_and_warmup(self):
        for change in ['step','nan','config','warmup']:
            with self.subTest(change=change):
                job,stage,directory=self.write_local()
                if change in ['step','nan']:
                    path=directory/'final_evaluation.json'
                    result=json.loads(path.read_text())
                    result['checkpoint_step' if change=='step' else 'episode_avg_return']=50000 if change=='step' else float('nan')
                    path.write_text(json.dumps(result))
                elif change=='config':
                    conf=stage_config(job,stage)
                    conf['JointTrainAgent']['Retrieval']['target']=16
                    (directory/'config.yaml').write_text(yaml.safe_dump(conf))
                else:(directory/'shared_warmup.json').write_text(json.dumps(dict(checkpoint_dir='/different/snapshot')))
                with self.assertRaises(ValueError):collector.local_result(self.root,job,stage,self.jobs[job['id']],self.manifest)

    def test_resumed_checkpoint_is_allowed_only_for_the_same_branch_and_original_plan(self):
        job,stage=self.job_stage()
        own=self.root/'ckpt'/stage['local_name']/'queue_resume_abcd'
        collector.check_source(str(own),job,stage,self.root,True)
        with self.assertRaises(ValueError):collector.check_source(str(own),job,stage,self.root,False)
        with self.assertRaises(ValueError):collector.check_source(str(own.parent.parent/'other'/'queue_resume_abcd'),job,stage,self.root,True)
        live=dict(self.jobs['j0'],plan={stage['key']:['--resume_from','/wrong']})
        with self.assertRaises(ValueError):collector.plan_proves_source(job,stage,live)

    def test_partial_warmups_do_not_change_comparison_or_heatmap_but_count_in_progress(self):
        rows=self.completed_rows({'j0'})
        partial=rows['j1','retrieval_h9_t4']
        partial.update(collector.evaluation_fields(999999,[999999]*20,'bf16',100000,20,0,self.manifest))
        data=report.aggregate(self.manifest,rows)
        self.assertEqual(data['common'],{'j0'})
        self.assertEqual(data['counts']['h9 / t4'],2)
        self.assertEqual(data['stats']['Alien','h9 / t4']['mean'],1100)
        self.assertEqual(data['stats']['Alien','h12 / t16']['mean'],1100)
        self.assertIsNone(data['stats']['Frostbite','h9 / t4']['mean'])
        self.assertTrue(data['provisional'])

    def test_overall_prioritizes_settings_and_interim_prioritizes_samples(self):
        rows = self.completed_rows(set())
        for identity, first, second in [('j0', 1000, 800), ('j1', 1200, 900), ('j2', 1500, 1400)]:
            rows[identity, 'retrieval_h9_t4'].update(State='completed', **{'Eval Return': first})
            rows[identity, 'retrieval_h10_t4'].update(State='completed', **{'Eval Return': second})
        rows['j3', 'retrieval_h9_t4'].update(State='completed', **{'Eval Return': 999999})
        rows['j0', 'retrieval_h11_t4'].update(State='completed', **{'Eval Return': 999999})
        rows['j0', 'baseline'].update(State='completed', **{'Eval Return': 500})

        data = report.aggregate(self.manifest, rows)
        self.assertEqual(data['common'], {'j0'})
        self.assertEqual(data['selected'], ('h9 / t4', 'h10 / t4', 'h11 / t4'))
        self.assertEqual(data['stats']['Alien', 'h9 / t4']['mean'], 1000)
        self.assertEqual(data['stats']['Alien', 'h9 / t4']['delta'], 500)
        interim = data['interim']
        self.assertEqual(interim['common'], {'j0', 'j1', 'j2'})
        self.assertEqual(interim['selected'], ('h9 / t4', 'h10 / t4'))
        self.assertIsNone(interim['stats']['Alien', 'h9 / t4']['delta'])
        self.assertIsNone(interim['metrics']['Baseline']['mean'])

        wb = report.build_workbook(self.manifest, data, self.manifest_path)
        self.assertEqual(wb.active.title, '전체 비교')
        ws = wb['중간 비교']
        self.assertEqual([ws['I5'].value, ws['J5'].value], [1100, 1500])
        self.assertEqual([ws['I6'].value, ws['J6'].value], [850, 1400])
        self.assertAlmostEqual(ws['E5'].value, (report.hns('Alien', 1100) + report.hns('Frostbite', 1500)) / 2)
        self.assertIn('사용 표본 3/4개', ws['A2'].value)
        self.assertEqual([ws['E11'].value, ws['F11'].value], [1000, 800])

    def test_comparison_ties_use_the_other_dimension(self):
        completed = {
            'A': frozenset({'j0', 'j1', 'j2'}),
            'B': frozenset({'j0', 'j1'}),
            'C': frozenset({'j0', 'j1'}),
            'D': frozenset({'j0', 'j2'}),
            'E': frozenset({'j2'}),
        }
        overall, interim = report.select_cohorts(completed, {'j0', 'j1', 'j2'})
        self.assertEqual(overall, ({'j0'}, ('A', 'B', 'C', 'D')))
        self.assertEqual(interim, ({'j0', 'j1'}, ('A', 'B', 'C')))
        completed.pop('D')
        completed['E'] = frozenset({'j2', 'j3'})
        completed['F'] = frozenset({'j2', 'j3'})
        overall, interim = report.select_cohorts(completed, {'j0', 'j1', 'j2', 'j3'})
        self.assertEqual(overall, ({'j0', 'j1'}, ('A', 'B', 'C')))
        self.assertEqual(interim, overall)

    def test_same_seed_on_different_gpus_is_a_separate_observation(self):
        data=report.aggregate(self.manifest,self.completed_rows({'j0','j3'}))
        self.assertEqual(len(data['common']),2)
        self.assertEqual(data['stats']['Alien','h9 / t4']['n'],2)

    def test_metric_conventions_cover_unequal_seed_counts_negative_hns_and_gap_clipping(self):
        metrics=report.calculate_metrics({'a':[0,4],'b':[.5,.5]})
        self.assertEqual(metrics,dict(mean=1.25,median=1.25,iqm=.5,gap=.25))
        metrics=report.calculate_metrics({'a':[-2,0,1,2,100]})
        self.assertEqual(metrics['iqm'],1)
        self.assertEqual(metrics['gap'],0)
        metrics=report.calculate_metrics({'a':[-2,0,1]})
        self.assertAlmostEqual(metrics['iqm'],-1/3)
        self.assertAlmostEqual(metrics['gap'],4/3)
        self.assertIsNone(report.calculate_metrics({})['mean'])

    def test_wandb_filters_exact_current_names_and_rejects_old_or_wrong_setting_runs(self):
        current=self.fake_run()
        old=self.fake_run(run_id='old')
        old.config['RunMetadata']['local_run_name']='Alien-old-study'
        queries=[]
        api=SimpleNamespace(runs=lambda project,**kwargs:(queries.append(kwargs) or [old,current]))
        remote,issues=collector.fetch_results(self.manifest,self.jobs,self.root,api=api)
        self.assertEqual(list(remote),[('j0','retrieval_h9_t4')])
        self.assertEqual(remote['j0','retrieval_h9_t4'][0]['Eval Return'],1000)
        self.assertFalse(issues)
        self.assertNotIn('Alien-old-study',queries[0]['filters']['config.RunMetadata.local_run_name']['$in'])
        bad=self.fake_run(run_id='bad')
        bad.config['JointTrainAgent']['Retrieval']['hash_bits']=12
        job,stage=self.job_stage()
        with self.assertRaisesRegex(ValueError,'configuration'):collector.classify_run(bad,job,stage,self.manifest,self.root)

    def test_failed_wandb_run_with_score_is_not_complete_and_retry_scores_are_not_averaged(self):
        job,stage=self.job_stage()
        failed=collector.classify_run(self.fake_run(state='crashed'),job,stage,self.manifest,self.root)
        self.assertEqual(failed['State'],'failed')
        self.assertNotIn('Eval Return',failed)
        first=collector.classify_run(self.fake_run(run_id='first',score=10),job,stage,self.manifest,self.root)
        latest=collector.classify_run(self.fake_run(run_id='latest',score=100),job,stage,self.manifest,self.root)
        latest['Created At']='2026-11-24T09:00:00+00:00'
        rows=collector.collect_rows(self.manifest,self.jobs,self.root,remote={('j0','retrieval_h9_t4'):[first,latest]})
        result=next(row for row in rows if row['Job ID']=='j0' and row['Stage']=='retrieval_h9_t4')
        self.assertEqual(result['Eval Return'],100)
        self.assertEqual(result['Attempt Count'],2)

    def test_foreign_or_duplicate_csv_is_rejected(self):
        row=dict(next(expected_rows(self.manifest))[2],State='pending')
        path=self.root/'input.csv'
        def write(rows):
            with path.open('w',newline='') as target:
                writer=csv.DictWriter(target,fieldnames=FIELDS)
                writer.writeheader();writer.writerows(rows)
        write([row,row])
        with self.assertRaisesRegex(ValueError,'Duplicate'):read_csv_rows(path,self.manifest)
        write([{**row,'Campaign ID':'previous-study'}])
        with self.assertRaisesRegex(ValueError,'another'):read_csv_rows(path,self.manifest)

    def test_final_wandb_score_requires_checkpoint_and_all_episode_returns(self):
        job,stage=self.job_stage()
        bad=self.fake_run()
        bad.summary['eval/checkpoint_step']=50000
        with self.assertRaises(ValueError):collector.classify_run(bad,job,stage,self.manifest,self.root)
        bad=self.fake_run()
        del bad.summary['eval/bf16/episode_return_19']
        with self.assertRaises(ValueError):collector.classify_run(bad,job,stage,self.manifest,self.root)

    def test_ctrl_c_finished_wandb_run_without_final_eval_keeps_queue_retry_pending(self):
        job,stage=self.job_stage()
        stopped=self.fake_run()
        stopped.summary={}
        candidate=collector.classify_run(stopped,job,stage,self.manifest,self.root)
        self.assertEqual(candidate['State'],'interrupted')
        self.assertNotIn('Eval Return',candidate)
        rows=collector.collect_rows(self.manifest,self.jobs,self.root,remote={('j0',stage['key']):[candidate]})
        result=next(r for r in rows if r['Job ID']=='j0' and r['Stage']==stage['key'])
        self.assertEqual(result['State'],'pending')
        self.assertIsNone(result['Eval Return'])

    def test_populated_workbook_has_global_metric_heatmaps_and_correct_common_game_scores(self):
        rows=self.completed_rows({'j0','j2','j3'})
        rows['j1','retrieval_h9_t4'].update(collector.evaluation_fields(999999,[999999]*20,'bf16',100000,20,0,self.manifest))
        data=report.aggregate(self.manifest,rows)
        wb=report.build_workbook(self.manifest,data,self.manifest_path)
        path=self.root/'complete.xlsx'
        wb.save(path)
        wb=load_workbook(path,data_only=True)
        ws=wb['전체 비교']
        first=next(row for row in ws.iter_rows(min_row=5,max_row=17,values_only=True) if row[2:4]==(9,4))
        self.assertEqual(first[4],'4 / 4')
        self.assertEqual(first[9:11],(1100,1600))
        self.assertAlmostEqual(ws['B23'].value,data['metrics']['h9 / t4']['mean'])
        self.assertAlmostEqual(ws['H23'].value,data['metrics']['h9 / t4']['median'])
        self.assertAlmostEqual(ws['B33'].value,data['metrics']['h9 / t4']['iqm'])
        self.assertAlmostEqual(ws['H33'].value,data['metrics']['h9 / t4']['gap'])
        self.assertIn('사용 표본 3/4개',ws['A3'].value)
        # The unmatched high score appears in the run details and progress count.
        self.assertEqual(data['stats']['Alien','h9 / t4']['n'],2)
        self.assertEqual(set(data['ranks'].values()),{1})

    def test_export_cells_do_not_interpret_notes_as_excel_formulas(self):
        rows=self.completed_rows(set())
        rows['j0','retrieval_h9_t4']['Notes']='=1+1'
        wb=report.build_workbook(self.manifest,report.aggregate(self.manifest,rows),self.manifest_path)
        cell=wb['실행 상세']['P7']
        self.assertEqual(cell.value,'=1+1')
        self.assertEqual(cell.data_type,'s')

    def test_named_experiments_infer_separate_grids_and_report_paths(self):
        for identity,name,pairs in [('j0','grid-a',GRID),('j1','grid-b',[[7,2],[14,32]])]:
            self.jobs[identity]['experiment_id']=name
            self.jobs[identity]['overrides'][PREFIX+'hash_target_pairs']=pairs
            with sqlite3.connect(self.database) as db:
                db.execute('UPDATE jobs SET payload=? WHERE id=?',(json.dumps(self.jobs[identity]),identity))
            output=collector.main(offline=True,root=self.root,experiment=name)
            manifest,csv_path,excel_path=report_paths(name,results_dir=self.root/'results')
            self.assertEqual(output,csv_path)
            cohort=load_manifest(manifest)
            self.assertEqual(cohort['pairs'],pairs)
            self.assertEqual([job['id'] for job in cohort['jobs']],[identity])
            self.assertEqual(len(read_csv_rows(output,cohort)),len(pairs)+1)
            report.main(csv_path,excel_path,manifest_path=manifest,experiment=name)
            self.assertTrue(excel_path.is_file())
        names=dict(list_experiments(read_jobs(self.database),self.root/'results'))
        self.assertEqual(names,{'grid-a':1,'grid-b':1})
        path=report_paths('grid-a',results_dir=self.root/'results')[0]
        before=path.read_bytes()
        # Queue jobs from another experiment never extend an already saved scope.
        collector.main(offline=True,root=self.root,experiment='grid-a')
        self.assertEqual(before,path.read_bytes())

    def test_named_report_rejects_new_jobs_reusing_an_already_frozen_name(self):
        self.jobs['j0']['experiment_id']='fixed'
        with sqlite3.connect(self.database) as db:
            db.execute('UPDATE jobs SET payload=? WHERE id=?',(json.dumps(self.jobs['j0']),'j0'))
        collector.main(offline=True,root=self.root,experiment='fixed')
        future=self.add_job('future','Alien','3090',7777)
        future['experiment_id']='fixed'
        with sqlite3.connect(self.database) as db:
            db.execute('UPDATE jobs SET payload=? WHERE id=?',(json.dumps(future),'future'))
        with self.assertRaisesRegex(ValueError,'reused'):
            collector.main(offline=True,root=self.root,experiment='fixed')

    def test_experiment_name_cannot_escape_the_report_directory(self):
        for name in ['../other','/tmp/other','','has space']:
            with self.assertRaises(ValueError):report_paths(name,results_dir=self.root)

    def test_offline_export_shows_one_setting_without_ranking_it(self):
        self.write_local(score=1234)
        # A historical same-game result has a different directory and cannot enter.
        old=self.root/'runs/Alien-old-study'
        old.mkdir(parents=True)
        (old/'final_evaluation.json').write_text(json.dumps(dict(episode_avg_return=999999)))
        csv_path=self.root/'grid.csv'
        output=self.root/'grid.xlsx'
        collector.main(output_path=csv_path,offline=True,manifest_path=self.manifest_path,root=self.root,excel_path=output)
        wb=load_workbook(output,data_only=True)
        ws=wb['전체 비교']
        self.assertEqual([cell.value for cell in ws[4]][:9],['순위','설정','hash_bits','target','완료 / 예정','Mean HNS','Median HNS','IQM','Optimality Gap'])
        self.assertEqual(ws.max_column,11)
        self.assertEqual(len(ws.conditional_formatting),4)
        self.assertEqual(len(ws._charts),0)
        self.assertIn('1/2게임 | 조합 간 순위 대기',ws['A3'].value)
        self.assertEqual([ws[cell].value for cell in ('A6','B6','E6','J6')],['—','h9 / t4','1 / 4',1234])
        self.assertAlmostEqual(ws['F6'].value,report.hns('Alien',1234))
        for row in ws.iter_rows(min_row=7,max_row=17,min_col=6,max_col=11):
            self.assertTrue(all(cell.value is None for cell in row))
        interim=wb['중간 비교']
        self.assertIn('조합 간 순위 대기',interim['A2'].value)
        self.assertEqual([interim[cell].value for cell in ('A5','B5','E10')],['—','h9 / t4',1234])
        details=list(wb['실행 상세'].iter_rows(min_row=7,values_only=True))
        self.assertEqual(len(details),52)
        self.assertEqual([row[7] for row in details if row[7] is not None],[1234])
        self.assertEqual(wb.sheetnames,['전체 비교','중간 비교','게임별 비교','GPU·시드별 점수','실행 상세','집계 기준'])


class NamedQueueTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        self.queue=TrainingQueue(self.root/'training_queue')
        self.spec=dict(game='Alien',seed=6000,gpu_type='A6000',config_text='config snapshot',
                       overrides={PREFIX+'hash_target_pairs':[[9,4]]},experiment_id='grid-a')

    def test_distinct_experiments_are_separate_but_repeating_one_is_idempotent(self):
        first,skipped=self.queue.enqueue_jobs([self.spec])
        self.assertEqual((len(first),skipped),(1,0))
        repeat,skipped=self.queue.enqueue_jobs([self.spec])
        self.assertEqual((repeat,skipped),([],1))
        other,skipped=self.queue.enqueue_jobs([{**self.spec,'experiment_id':'grid-b'}])
        self.assertEqual((len(other),skipped),(1,0))
        self.assertNotEqual(first[0]['id'],other[0]['id'])
        self.queue.finish(first[0]['id'],'completed')
        repeat,skipped=self.queue.enqueue_jobs([self.spec])
        self.assertEqual((repeat,skipped),([],1))

    def test_existing_name_cannot_be_reused_for_another_grid_or_membership(self):
        self.queue.enqueue_jobs([self.spec])
        altered=deepcopy(self.spec)
        altered['overrides'][PREFIX+'hash_target_pairs']=[[12,16]]
        with self.assertRaisesRegex(ValueError,'another job set'):self.queue.enqueue_jobs([altered])
        with self.assertRaisesRegex(ValueError,'another job set'):
            self.queue.enqueue_jobs([self.spec,{**self.spec,'game':'Pong'}])
        self.assertEqual(len(self.queue.jobs(statuses=None)),1)

    def test_removed_named_jobs_do_not_get_silently_recreated(self):
        self.queue.enqueue_jobs([self.spec])
        with self.queue.connect() as db:db.execute('DELETE FROM jobs')
        with self.assertRaisesRegex(ValueError,'removed'):self.queue.enqueue_jobs([self.spec])

    def test_saved_report_alias_reserves_its_name_without_modifying_running_jobs(self):
        path=self.root/'results/experiments/grid-a/manifest.json'
        path.parent.mkdir(parents=True)
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'saved report'):self.queue.enqueue_jobs([self.spec])
        self.assertEqual(self.queue.jobs(statuses=None),[])


if __name__=='__main__':
    unittest.main()
