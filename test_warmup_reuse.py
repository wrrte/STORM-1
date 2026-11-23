"""Compatibility, automatic discovery, replacement cleanup and resumable branch plans."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from queue_train import ROOT, PAIR_PREFIX, TrainingQueue, read_seed_files, save_seed_files
from enqueue_train import build_specs
from warmup_catalog import (request_config, compatibility, choose_warmup, branch_commands,
                            discover, prune_replaced)
from queue_job import JobRunner, resume_command
import queue_checkpoint


class WarmupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.queue = TrainingQueue(self.root / 'training_queue')
        self.spec = build_specs([['Alien', 6000]], 'A6000', ROOT / 'config_files/STORM.yaml', {
            PAIR_PREFIX+'warmup_offset_pairs': [], PAIR_PREFIX+'hash_target_pairs': [[9,4],[10,8]],
            PAIR_PREFIX+'enable': 'Both'}, 0)[0]
        self.job = self.queue.enqueue_jobs([self.spec])[0][0]
        self.record = self.make_record('old')

    def tearDown(self):
        self.temp.cleanup()

    def make_record(self, run, signals=('td_error', 'value_diff', 'value')):
        source = self.root / 'ckpt' / run / 'shared_warmup_50001'
        source.mkdir(parents=True)
        conf = request_config(self.job)
        conf.JointTrainAgent.Retrieval.warmup_offset_pairs = []
        conf.JointTrainAgent.Retrieval.hash_target_pairs = []
        conf.JointTrainAgent.Retrieval.save_warmup = True
        meta = dict(version=1, run_name=run, seed=6000, env_name='ALE/Alien-v5', step=50001,
                    trajectory_path=str(self.root/'D_TRAJ/Alien.pkl'), cwd=str(self.root),
                    statistics_signals=list(signals), requested_warmup_steps=[50000])
        (source/'config.yaml').write_text(conf.dump())
        (source/'warmup_metadata.json').write_text(json.dumps(meta))
        (source/'training_state.pt').write_text('test model payload')
        (source/'replay_buffer.npz').write_text('test replay payload')
        return dict(source=str(source.relative_to(self.root)), resume_warmup=str(source),
                    config=conf.dump(), metadata=meta, legacy=False, gpu_type='A6000',
                    game='Alien', seed=6000, step=50001, warmup_steps=[50000], statistics_signals=list(signals))

    def test_retrieval_sweeps_reuse_but_prefix_changes_do_not(self):
        self.assertIsNone(compatibility(self.record, self.job))
        for key, value, reason in [('Models.Agent.HiddenDim', 256, 'Models'),
                                   (PAIR_PREFIX+'ema_alpha', .02, 'ema_alpha'),
                                   (PAIR_PREFIX+'warmup_steps', 30000, 'warmup target'),
                                   ('JointTrainAgent.UseDemonstration', True, 'UseDemonstration')]:
            job=copy.deepcopy(self.job)
            job['overrides'][key]=value
            self.assertIn(reason, compatibility(self.record, job))
        job=copy.deepcopy(self.job)
        job['overrides'][PAIR_PREFIX+'anchor_offset']=0
        job['overrides'][PAIR_PREFIX+'hash_bits']=12
        self.assertIsNone(compatibility(self.record, job))

    def test_missing_statistics_need_replacement(self):
        self.record['statistics_signals']=['td_error','value_diff']
        self.job['overrides'][PAIR_PREFIX+'enable']=['value']
        record, reasons=choose_warmup(self.job,dict(available=[self.record]))
        self.assertIsNone(record)
        self.assertIn('missing statistics',reasons[0]['reason'])

    def test_existing_warmup_plan_contains_all_hash_arms_and_one_baseline(self):
        commands=branch_commands(self.job,self.record,root=self.root)
        self.assertEqual(set(commands),{'retrieval_h9_t4','retrieval_h10_t8','baseline'})
        for command in commands.values():
            self.assertIn('--resume_from',command)
            self.assertEqual(command[command.index('--resume_from')+1],self.record['resume_warmup'])
            self.assertEqual(command[0],self.job['python'])
        # Freeze request values even if its original YAML changes later.
        self.assertEqual(request_config(self.job).JointTrainAgent.Retrieval.hash_target_pairs,[[9,4],[10,8]])

    def test_discovery_connects_new_snapshots_even_for_former_fresh_entry(self):
        save_seed_files({'alien:6000':dict(game='Alien',seed=6000,gpu_type='A6000',fresh=True,
                                         enabled=False,warmups=[])},self.queue.seed_directory)
        with patch('warmup_catalog.inspect_warmup',return_value=copy.deepcopy(self.record)) as inspect:
            report=discover('A6000',root=self.root,queue_dir=self.queue.directory,seed_dir=self.queue.seed_directory)
            self.assertEqual(len(report['available']),1)
            selection=read_seed_files(self.queue.seed_directory)['alien:6000']
            self.assertFalse(selection['enabled'])
            self.assertEqual(selection['warmups'],[self.record['source']])
            self.assertNotIn('fresh',selection)
            discover('A6000',root=self.root,queue_dir=self.queue.directory,seed_dir=self.queue.seed_directory)
            self.assertEqual(inspect.call_count,1) # Unchanged huge payloads are not read again.
            new=self.make_record('new')
            inspect.return_value=new
            report=discover('A6000',root=self.root,queue_dir=self.queue.directory,seed_dir=self.queue.seed_directory)
            self.assertEqual(len(report['available']),2)
            self.assertEqual(len(read_seed_files(self.queue.seed_directory)['alien:6000']['warmups']),2)

    def test_dry_discovery_does_not_write_catalog_or_selections(self):
        with patch('warmup_catalog.inspect_warmup',return_value=self.record):
            discover('A6000',root=self.root,queue_dir=self.queue.directory,seed_dir=self.queue.seed_directory,persist=False)
        self.assertFalse((self.queue.directory/'warmup_inventory.json').exists())
        self.assertFalse(self.queue.seed_directory.exists())

    def test_prune_waits_for_replacement_and_preserves_pinned_or_useful_sources(self):
        replacement=self.make_record('replacement')
        self.record['statistics_signals']=['td_error','value_diff']
        report=dict(available=[self.record,replacement])
        (self.queue.directory/'warmup_inventory.json').write_text(json.dumps(report))
        incompatible=[dict(source=self.record['source'],reason='missing statistics')]
        # Another experiment still benefits from the old source.
        other=copy.deepcopy(self.spec)
        other['overrides'][PAIR_PREFIX+'enable']=['baseline']
        otherjob=self.queue.enqueue_jobs([other])[0][0]
        deleted=prune_replaced(self.job,replacement,incompatible,self.queue,root=self.root)
        self.assertEqual(deleted,[])
        self.queue.finish(otherjob['id'],'completed')
        deleted=prune_replaced(self.job,replacement,incompatible,self.queue,root=self.root)
        self.assertEqual(deleted,[self.record['source']])
        self.assertFalse((self.root/self.record['source']).exists())
        self.assertTrue((self.root/replacement['source']).exists())

    def test_failed_job_keeps_its_pinned_source_for_retry(self):
        replacement=self.make_record('replacement')
        other=copy.deepcopy(self.spec)
        other['overrides']['Models.Agent.HiddenDim']=256
        other['warmup_source']=self.record['source']
        failed=self.queue.enqueue_jobs([other])[0][0]
        self.queue.finish(failed['id'],'failed',1)
        (self.queue.directory/'warmup_inventory.json').write_text(json.dumps(dict(available=[self.record,replacement])))
        deleted=prune_replaced(self.job,replacement,[dict(source=self.record['source'],reason='model')],self.queue,root=self.root)
        self.assertEqual(deleted,[])
        self.assertTrue((self.root/self.record['source']).exists())

    def test_other_warmup_targets_are_not_deleted(self):
        replacement=self.make_record('replacement')
        self.record['warmup_steps']=[30000]
        (self.queue.directory/'warmup_inventory.json').write_text(json.dumps(dict(available=[self.record,replacement])))
        deleted=prune_replaced(self.job,replacement,[dict(source=self.record['source'],reason='target')],self.queue,root=self.root)
        self.assertEqual(deleted,[])

    def test_runner_collects_only_when_needed_and_persists_branch_completion(self):
        empty=dict(available=[])
        populated=dict(available=[self.record])
        runner=JobRunner(self.queue,self.job,root=self.root)
        calls=[]
        def execute(command,stage,**kwargs):
            calls.append(stage)
            if stage=='collect':
                from yacs.config import CfgNode
                conf=CfgNode.load_cfg(Path(command[command.index('-config_path')+1]).read_text())
                self.assertTrue(conf.JointTrainAgent.Retrieval.save_warmup)
                self.assertEqual(conf.JointTrainAgent.Retrieval.hash_target_pairs,[])
            return 0
        with patch('warmup_catalog.discover',side_effect=[empty,populated]), \
             patch('warmup_catalog.prune_replaced',return_value=[]), patch.object(runner,'execute',side_effect=execute):
            self.assertEqual(runner.run(),0)
        self.assertEqual(calls,['collect','retrieval_h9_t4','retrieval_h10_t8','baseline'])
        saved=self.queue.jobs()[0]
        self.assertEqual(saved['completed_branches'],calls[1:])
        second=JobRunner(self.queue,saved,root=self.root)
        with patch('warmup_catalog.discover',side_effect=AssertionError('must use persisted plan')), \
             patch.object(second,'execute',side_effect=AssertionError('completed branch reran')):
            self.assertEqual(second.run(),0)

    def test_runner_uses_existing_source_without_collection(self):
        runner=JobRunner(self.queue,self.job,root=self.root)
        with patch('warmup_catalog.discover',return_value=dict(available=[self.record])), \
             patch('warmup_catalog.prune_replaced',return_value=[]), patch.object(runner,'execute',return_value=0) as execute:
            self.assertEqual(runner.run(),0)
        self.assertNotIn('collect',[call.args[1] for call in execute.call_args_list])

    def test_resume_checkpoint_is_published_only_after_save_and_cleaned_on_success(self):
        old_cwd=Path.cwd()
        os.chdir(self.root)
        try:
            def save(path):
                p=Path(path);p.mkdir()
                (p/'training_state.pt').write_text('model')
                (p/'replay_buffer.npz').write_text('replay')
            queue_checkpoint.save_resume('test_baseline',save)
            locator=Path('ckpt/test_baseline/queue_resume.json')
            previous=locator.read_text()
            with self.assertRaises(RuntimeError):
                queue_checkpoint.save_resume('test_baseline',lambda _: (_ for _ in ()).throw(RuntimeError('interrupted write')))
            self.assertEqual(locator.read_text(),previous)
            command=[sys.executable,'train.py','-n','test','--branch_experiment','baseline','--resume_from','original']
            resumed=resume_command(command,root=self.root)
            self.assertNotEqual(resumed[-1],'original')
            with patch.dict(os.environ,STORM_QUEUE_JOB='test',STORM_QUEUE_STAGE='test:baseline'):
                queue_checkpoint.stage_complete('test_baseline')
            self.assertFalse(locator.exists())
            self.assertEqual(list(Path('ckpt/test_baseline').glob('queue_resume_*')),[])
        finally:
            os.chdir(old_cwd)


if __name__=='__main__':
    unittest.main()
