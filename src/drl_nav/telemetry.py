# telemetry.py — Telemetry Logger
# NOVELTY 4: Tracks everything across all episodes.
# This gives you PROOF that the robot is learning.
# success_rate going up + collision_rate going down = working.

from dataclasses import dataclass, field
from collections import deque
import time

@dataclass
class Telemetry:
    rewards: list = field(default_factory=list)
    success: list = field(default_factory=list)
    collisions: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=50))

    def log_episode(self, reward, success, collision, steps):
        """Call this at the end of every episode."""
        self.rewards.append(float(reward))
        self.success.append(int(success))
        self.collisions.append(int(collision))
        self.steps.append(int(steps))
        self.timestamps.append(time.time())

    def summary(self):
        """Returns current stats — use this for graphs and reports."""
        n = max(len(self.rewards), 1)
        return {
            "episodes_run": n,
            "mean_reward": round(sum(self.rewards) / n, 2),
            "success_rate": round(sum(self.success) / n * 100, 1),   # percentage
            "collision_rate": round(sum(self.collisions) / n * 100, 1),
            "mean_steps": round(sum(self.steps) / n, 1),
        }