"""Track a reward-triggered warmup without changing the training RNG."""
import math


class DynamicWarmup:
    def __init__(self, config, detected_step=-1):
        self.min_steps = config.get("min_warmup_steps", 0)
        self.target_steps = config.get("dynamic_warmup_target_steps", 68500)
        self.max_steps = config.get("max_warmup_steps", 80000)
        self.delay_steps = config.get("dynamic_warmup_delay_steps", 0)
        self.multiplier = config.get("dynamic_warmup_multiplier", 2.6)
        for name, value in self.settings.items():
            if name == "dynamic_warmup_multiplier":
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError(f"Retrieval.{name} must be a finite positive number")
            elif type(value) is not int or value < 0:
                raise ValueError(f"Retrieval.{name} must be a nonnegative integer")
        if type(detected_step) is not int or detected_step < -1:
            raise ValueError("Dynamic warmup detected_step must be -1 or a nonnegative integer")
        self.detected_step = detected_step
        self._checked_rewards = False

    @property
    def settings(self):
        return {
            "min_warmup_steps": self.min_steps,
            "dynamic_warmup_target_steps": self.target_steps,
            "max_warmup_steps": self.max_steps,
            "dynamic_warmup_delay_steps": self.delay_steps,
            "dynamic_warmup_multiplier": self.multiplier,
        }

    @property
    def end_step(self):
        if self.detected_step == -1:
            return self.max_steps
        return min(self.max_steps, max(self.target_steps,
                                       self.detected_step + self.delay_steps))

    def update(self, step, episode_rewards, detector, *, rewards_changed=False):
        """Return target readiness; the caller still waits for an episode boundary.

        Check once on becoming eligible (also after resume), then only when a
        completed episode changes the reward history. Detection stays latched.
        """
        if (step < self.max_steps and step >= self.min_steps and self.detected_step == -1
                and (rewards_changed or not self._checked_rewards)):
            self._checked_rewards = True
            if len(episode_rewards) and detector(episode_rewards, multiplier=self.multiplier):
                self.detected_step = step
        return step >= self.end_step

    def state_dict(self):
        return {"detected_step": self.detected_step, "target_step": self.end_step,
                "settings": self.settings}
