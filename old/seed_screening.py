"""Optional early seed screening without changing training randomness."""

import json
import math
from pathlib import Path


SCREEN_REJECTED_EXIT_CODE = 75


def configure_seed_screen(conf, args, branch_commands):
    """Screen a fresh baseline or the common prefix of fixed-warmup branches.

    A shared prefix is screened once. Its resumed children must not screen
    again: they start beyond the deadline and intentionally diverge afterward.
    """
    steps = conf.JointTrainAgent.SeedScreenSteps
    if steps == 0:
        return None, branch_commands
    retrieval = conf.JointTrainAgent.Retrieval
    shared = isinstance(branch_commands, (dict, tuple))
    if (conf.Task != "JointTrainAgent" or args.resume_from
            or conf.JointTrainAgent.NumEnvs != 1
            or (branch_commands is not None and not shared)
            or (not shared and retrieval.enable is not False)):
        raise ValueError("Seed screening requires a fresh baseline or fixed shared warmup with NumEnvs=1")
    if steps > conf.JointTrainAgent.SampleMaxSteps:
        raise ValueError("SeedScreenSteps cannot exceed SampleMaxSteps")
    if shared:
        if retrieval.warmup_steps < steps:
            raise ValueError("Shared warmup must reach SeedScreenSteps before branching")
        def child(command):
            return [*command, "JointTrainAgent.SeedScreenSteps", "0"]
        branch_commands = ({name: child(command) for name, command in branch_commands.items()}
                           if isinstance(branch_commands, dict)
                           else tuple(child(command) for command in branch_commands))
    screen = SeedScreen(steps, conf.JointTrainAgent.SeedScreenMinReturn,
                        f"runs/{args.n}", args.seed, args.n)
    return screen, branch_commands


class SeedScreen:
    def __init__(self, steps, min_return, run_dir, seed, run_name):
        if type(steps) is not int or steps <= 0:
            raise ValueError("SeedScreenSteps must be a positive integer")
        if not math.isfinite(min_return):
            raise ValueError("SeedScreenMinReturn must be finite")
        self.steps = steps
        self.min_return = float(min_return)
        self.run_dir = Path(run_dir)
        self.seed = seed
        self.run_name = run_name
        self.best_return = None
        self.passed = None
        self.decision_step = None

    def observe(self, step, episode_returns):
        """Observe cumulative returns before terminal episodes are reset.

        A success at exactly the deadline counts. A return equal to the
        threshold does not count. Once decided, later returns cannot change it.
        """
        if self.passed is not None:
            return None
        if step > self.steps:
            raise ValueError("Seed screening must observe every environment step through its deadline")
        current_best = max(float(value) for value in episode_returns)
        if not math.isfinite(current_best):
            raise ValueError("Non-finite return during seed screening")
        self.best_return = current_best if self.best_return is None else max(self.best_return, current_best)
        if self.best_return > self.min_return:
            self.passed = True
        elif step == self.steps:
            self.passed = False
        else:
            return None
        self.decision_step = step
        return self.passed

    def record(self, logger):
        if self.passed is None:
            raise ValueError("Seed screening has not reached a decision")
        result = {
            "status": "accepted" if self.passed else "rejected",
            "seed": self.seed,
            "run_name": self.run_name,
            "screen_steps": self.steps,
            "min_return_exclusive": self.min_return,
            "best_episode_return": self.best_return,
            "decision_step": self.decision_step,
            "includes_unfinished_episodes": True,
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "seed_screen.json").write_text(json.dumps(result, indent=2) + "\n")
        for key in ("screen_steps", "best_episode_return", "decision_step"):
            logger.log(f"seed_screen/{key}", result[key])
        logger.log("seed_screen/min_return_exclusive", self.min_return)
        logger.log("seed_screen/accepted", int(self.passed))
        print(f"Seed {self.seed} {result['status']} at step {self.decision_step}: "
              f"best episode score {self.best_return:g}; required > {self.min_return:g} "
              f"by step {self.steps}.", flush=True)
