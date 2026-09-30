"""Run the shared tests with their old STORM/ path mapped in a temporary fixture."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='storm-shared-regressions-') as directory:
        fixture = Path(directory)
        shutil.copytree(root / 'tests', fixture / 'tests', ignore=shutil.ignore_patterns('__pycache__'))
        for item in root.glob('*.py'):
            if item.name == 'training_branches.py':
                # This module resolves its own path when launching the supervisor.
                shutil.copy2(item, fixture / item.name)
            else:
                (fixture / item.name).symlink_to(item)
        for name in ('STORM-1', 'Drama'):
            (fixture / name).symlink_to(root / name, target_is_directory=True)
        (fixture / 'STORM').symlink_to(root / 'STORM-1', target_is_directory=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        return subprocess.call([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                                '-p', 'test_retrieval*.py', '-v'], cwd=fixture, env=env)


if __name__ == '__main__':
    sys.exit(main())
