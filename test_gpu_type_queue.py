"""Queue invariants and process recovery without requiring an NVIDIA device."""
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from queue_train import TrainingQueue, gpu_type, read_seed_files, save_seed_files


def drain(directory, output):
    queue = TrainingQueue(directory)
    while True:
        job, lease = queue.claim(dict(gpu_type='A6000', uuid=str(os.getpid()), host='test', index='4'))
        if not job:
            break
        queue.finish(job['id'], 'completed', 0)
        lease.close()
        output.put(job['id'])


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.queue = TrainingQueue(self.root / 'training_queue')
        self.device = dict(gpu_type='A6000', index='0', uuid='GPU-0', host='test')

    def tearDown(self):
        self.temp.cleanup()

    def add(self, seed=6000, model='A6000', priority=0, **extra):
        return self.queue.enqueue_jobs([dict(game='Alien', seed=seed, gpu_type=model,
            config_text='test', overrides={}, priority=priority, **extra)])[0][0]

    def test_model_aliases_and_distinct_types(self):
        self.assertEqual(gpu_type('NVIDIA RTX A6000'), 'A6000')
        self.assertEqual(gpu_type('NVIDIA GeForce RTX 3090'), '3090')
        self.assertEqual(gpu_type('NVIDIA RTX PRO 6000 Blackwell Workstation Edition'), gpu_type('pro6k'))
        self.assertNotEqual(gpu_type('RTX 6000 Ada Generation'), 'A6000')
        with self.assertRaises(ValueError):
            gpu_type('0')

    def test_seed_ownership_is_independent_of_game_and_index(self):
        self.add()
        self.queue.enqueue_jobs([dict(game='Pong', seed=6000, gpu_type='A6000', config_text='test', overrides={})])
        with self.assertRaisesRegex(ValueError, 'belongs to A6000'):
            self.queue.enqueue_jobs([dict(game='Pong', seed=6000, gpu_type='3090', config_text='test', overrides={})])
        device = dict(self.device, index='17', uuid='GPU-17')
        job, lease = self.queue.claim(device)
        self.assertEqual(job['seed'], 6000)
        lease.close()

    def test_global_priority_and_fifo(self):
        low = self.add(1)
        first = self.add(2, priority=10)
        second = self.add(3, priority=10)
        self.add(4, model='3090', priority=100)
        for expected in (first, second, low):
            job, lease = self.queue.claim(dict(self.device, index='7'))
            self.assertEqual(job['id'], expected['id'])
            self.queue.finish(job['id'], 'completed')
            lease.close()
        self.assertEqual(self.queue.claim(self.device), (None, None))

    def test_live_pair_is_serialized_and_running_job_is_deduplicated(self):
        first = self.add()
        job, lease = self.queue.claim(self.device)
        added, skipped = self.queue.enqueue_jobs([dict(game='Alien', seed=6000, gpu_type='A6000',
                                                      config_text='test', overrides={})])
        self.assertEqual((added, skipped), ([], 1))
        self.queue.enqueue_jobs([dict(game='Alien', seed=6000, gpu_type='A6000', config_text='different', overrides={})])
        self.assertEqual(self.queue.claim(dict(self.device, index='4')), (None, None))
        self.queue.finish(first['id'], 'completed')
        lease.close()
        job, lease = self.queue.claim(dict(self.device, index='4'))
        self.assertEqual(job['config_text'], 'different')
        lease.close()

    def test_published_warmup_allows_same_pair_on_multiple_cards(self):
        self.add()
        first, lease = self.queue.claim(self.device)
        first['plan'] = {'baseline': ['example']}
        self.queue.update_job(first)
        self.queue.enqueue_jobs([dict(game='Alien', seed=6000, gpu_type='A6000', config_text='other', overrides={})])
        second, second_lease = self.queue.claim(dict(self.device, index='7'))
        self.assertIsNotNone(second)
        self.assertNotEqual(first['id'], second['id'])
        lease.close()
        second_lease.close()

    def test_retry_preserves_progress_and_priority(self):
        job = self.add(priority=15, completed_branches=['baseline'])
        self.queue.finish(job['id'], 'failed', 1)
        self.queue.retry(job['id'])
        pending = self.queue.jobs()[0]
        self.assertEqual(pending['completed_branches'], ['baseline'])
        self.assertEqual(pending['priority'], 15)
        with self.assertRaises(ValueError):
            self.queue.retry(job['id'])

    def test_interrupted_job_moves_between_cards_with_progress(self):
        first = self.add()
        job, lease = self.queue.claim(self.device)
        job['completed_branches'] = ['baseline']
        self.queue.update_job(job)
        self.queue.finish(job['id'], 'pending', 143)
        lease.close()
        job, lease = self.queue.claim(dict(self.device, index='7', uuid='GPU-7'))
        self.assertEqual(job['id'], first['id'])
        self.assertEqual(job['completed_branches'], ['baseline'])
        lease.close()

    def test_abandoned_claim_recovers_only_after_last_inherited_lock_closes(self):
        first = self.add()
        job, lease = self.queue.claim(self.device)
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], pass_fds=(lease.fileno(),))
        lease.close()
        try:
            self.assertEqual(self.queue.recover(), [])
            self.assertEqual(self.queue.claim(dict(self.device, index='4')), (None, None))
        finally:
            child.terminate()
            child.wait(timeout=5)
        self.assertEqual(self.queue.recover(), [first['id']])
        job, lease = self.queue.claim(dict(self.device, index='4'))
        self.assertEqual(job['id'], first['id'])
        lease.close()

    def test_claims_are_atomic_across_processes(self):
        expected = {self.add(seed)['id'] for seed in range(20)}
        ctx = multiprocessing.get_context('fork')
        output = ctx.Queue()
        children = [ctx.Process(target=drain, args=(self.queue.directory, output)) for _ in range(4)]
        for child in children:
            child.start()
        for child in children:
            child.join(15)
            self.assertEqual(child.exitcode, 0)
        actual = [output.get(timeout=3) for _ in range(20)]
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(set(actual)))

    def test_seed_allocation_and_conflicting_selection_file(self):
        first = self.queue.allocate_seeds('A6000', 3)
        second = self.queue.allocate_seeds('3090', 3)
        self.assertFalse(set(first) & set(second))
        self.assertEqual(first, [1000000, 1000001, 1000002])
        self.assertEqual(second, [2000000, 2000001, 2000002])
        preview = self.queue.allocate_seeds('A6000', 2, reserve=False)
        self.assertEqual(preview, self.queue.allocate_seeds('A6000', 2, reserve=False))
        with self.assertRaisesRegex(ValueError, 'reserved A6000'):
            self.add(seed=1000500, model='3090')
        selections = {'a': dict(game='Alien', seed=10, gpu_type='A6000'),
                      'b': dict(game='Pong', seed=10, gpu_type='3090')}
        save_seed_files(selections, self.queue.seed_directory)
        with self.assertRaisesRegex(ValueError, 'multiple GPU types'):
            read_seed_files(self.queue.seed_directory)

    def test_migration_keeps_old_files_and_restores_interrupted_jobs(self):
        seed_dir = self.queue.seed_directory
        seed_dir.mkdir(parents=True)
        original = json.dumps(dict(gpu='0', game_seeds=[dict(game='Alien', seed=6000, warmups=['ckpt/old'])]))
        (seed_dir / 'gpu_0.json').write_text(original)
        config = self.root / 'config.yaml'
        config.write_text((Path(__file__).parent / 'config_files/STORM.yaml').read_text())
        job = dict(game='Alien', seed=6000, gpu='0', config=str(config), overrides={'JointTrainAgent.Retrieval.warmup_offset_pairs': [], 'JointTrainAgent.Retrieval.hash_target_pairs': []}, id='legacy')
        (self.queue.directory / 'history.jsonl').write_text(json.dumps(dict(event='interrupted', job=job)) + '\n')
        self.assertEqual(self.queue.migrate('A6000'), 1)
        self.assertEqual(self.queue.migrate('A6000'), 0)
        self.assertEqual((seed_dir / 'gpu_0.json').read_text(), original)
        self.assertTrue((self.queue.directory / 'legacy_backup' / 'gpu_0.json').is_file())
        self.assertEqual(read_seed_files(seed_dir)['alien:6000']['gpu_type'], 'A6000')
        self.assertEqual(self.queue.jobs()[0]['id'], 'legacy')

    def test_worker_shutdown_then_another_index_finishes_only_remaining_branch(self):
        began = self.root / 'began'
        finished = self.root / 'finished'
        code = ("from pathlib import Path; import sys,time; "
                f"p=Path({str(began)!r}); already=p.exists(); p.write_text('started'); "
                "time.sleep(0.2 if already else 30); "
                f"Path({str(finished)!r}).write_text('finished')")
        plan = {'done': [sys.executable, '-c', "raise RuntimeError('completed branch reran')", '-n', 'test', '--branch_experiment', 'baseline'],
                'remaining': [sys.executable, '-c', code, '-n', 'test', '--branch_experiment', 'baseline']}
        job = self.add(plan=plan, completed_branches=['done'])
        def worker(index):
            script = ("import queue_train as q; from pathlib import Path; "
                      "Original=q.TrainingQueue; "
                      f"q.TrainingQueue=lambda:Original({str(self.queue.directory)!r}); "
                      f"q.detect_device=lambda _:dict(gpu_type='A6000',index={index!r},uuid='GPU-{index}',host='test'); "
                      f"raise SystemExit(q.run_worker({index!r},once=True))")
            return subprocess.Popen([sys.executable, '-c', script], cwd=Path(__file__).parent,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        first = worker('0')
        try:
            deadline = time.monotonic() + 8
            while not began.exists() and first.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertTrue(began.exists(), first.stderr.read().decode() if first.poll() is not None else 'trainer did not start')
            first.send_signal(signal.SIGTERM)
            _, errors = first.communicate(timeout=10)
            self.assertEqual(first.returncode, 143, errors.decode())
            self.assertEqual(self.queue.jobs()[0]['status'], 'pending')
            second = worker('7')
            _, errors = second.communicate(timeout=10)
            self.assertEqual(second.returncode, 0, errors.decode())
            self.assertTrue(finished.exists())
            completed = self.queue.jobs(statuses=('completed',))[0]
            self.assertEqual(completed['completed_branches'], ['done', 'remaining'])
        finally:
            if first.poll() is None:
                first.terminate()
                first.wait(timeout=10)
            if first.stderr:
                first.stderr.close()


if __name__ == '__main__':
    unittest.main()
