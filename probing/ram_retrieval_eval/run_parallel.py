"""Run Seaquest, Frostbite and Pong concurrently; seeds stay sequential per game.

python probing/ram_retrieval_eval/run_parallel.py --devices cuda:0 \
    --steps 100000 --buffer-size 20000 --seeds 2027 2028 2029 2030 2031 \
    --random-repeats 5 --bootstrap-samples 5000 \
    --output results/ram_verification/retrieval_9999_parallel

Pass three --devices values to assign one device per game, in the order above.
All other arguments are forwarded to evaluate.py. --dry-run prints commands only.
"""
import argparse
from contextlib import ExitStack
from pathlib import Path
import shlex
import subprocess
import sys

try:
    from .evaluate import GAMES, ROOT, parse_args
except ImportError:
    from evaluate import GAMES, ROOT, parse_args


def stop_children(children):
    for _, process in children:
        if process.poll() is None:
            process.terminate()
    for _, process in children:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"],
                        help="One shared device, or three devices for seaquest/frostbite/pong")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/ram_verification/retrieval_9999_parallel")
    parser.add_argument("--dry-run", action="store_true")
    launch, forwarded = parser.parse_known_args(argv)
    if len(launch.devices) not in (1, len(GAMES)):
        parser.error("--devices requires one shared device or exactly three devices")
    for argument in forwarded:
        if argument.split("=", 1)[0] in ("--game", "--device", "--checkpoint", "--config"):
            parser.error(f"{argument.split('=', 1)[0]} is not supported by the three-game launcher; use --devices for device assignment")

    # Validate evaluator options and check all destinations before starting jobs.
    evaluation = parse_args(["--output", str(launch.output), *forwarded])
    if "auto" in launch.devices:
        parser.error("Use an explicit device, such as cuda:0 or cpu")
    devices = launch.devices * len(GAMES) if len(launch.devices) == 1 else launch.devices
    seeds = evaluation.seeds or [evaluation.seed]
    for game in GAMES:
        for seed in seeds:
            destination = evaluation.output / game
            if evaluation.seeds:
                destination /= f"seed_{seed}"
            if destination.exists():
                parser.error(f"{destination} already exists; choose a new --output")

    jobs = []
    for game, device in zip(GAMES, devices):
        command = [sys.executable, "-u", str(Path(__file__).with_name("evaluate.py")),
                   "--output", str(evaluation.output), *forwarded,
                   "--game", game, "--device", device]
        log = evaluation.output / "logs" / f"{game}.log"
        jobs.append((game, command, log))
        print(f"[{game}] {shlex.join(command)}", flush=True)
        print(f"[{game}] log: {log}", flush=True)
    if launch.dry_run:
        return 0

    children = []
    with ExitStack() as files:
        try:
            for game, command, log in jobs:
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = files.enter_context(log.open("x"))
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                children.append((game, process))
                print(f"[{game}] started PID {process.pid}", flush=True)
            failed = []
            for game, process in children:
                code = process.wait()
                print(f"[{game}] finished with exit code {code}", flush=True)
                if code:
                    failed.append(game)
            if failed:
                print("Failed games (see their logs): " + ", ".join(failed), file=sys.stderr)
            return int(bool(failed))
        except KeyboardInterrupt:
            print("Stopping all three game processes...", file=sys.stderr)
            return 130
        finally:
            stop_children(children)


if __name__ == "__main__":
    sys.exit(main())
