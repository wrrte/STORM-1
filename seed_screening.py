"""Optional early seed screening without changing training randomness."""

import json
import math
from pathlib import Path


SCREEN_REJECTED_EXIT_CODE = 75


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
