# env.py — The Robot's World
# This file defines:
# 1. What the robot sees (LiDAR + goal distance + goal angle)
# 2. How the robot moves (linear and angular velocity)
# 3. Whether an episode ended (collision / success / timeout)

import math
import random
import numpy as np
from collections import deque

def adaptive_reward(goal_dist, progress, min_lidar, linear, angular, step_count):
    """
    NOVELTY 1: Adaptive Reward Shaping
    Instead of fixed rewards, this changes based on context.
    - Moving toward goal? Good reward.
    - Near a wall? Heavy penalty.
    - Spinning too much? Small penalty.
    - Taking too long? Time pressure.
    """
    reward = 0.0
    reward += 15.0 * progress           # moved closer to goal = good
    reward -= 0.01 * goal_dist          # still far from goal = small penalty
    reward -= 0.03 * abs(angular)       # spinning too much = bad
    reward -= 0.005 * step_count        # taking too long = bad

    # If close to obstacle, penalty gets STRONGER (this is the adaptive part)
    if min_lidar < 0.6:
        reward -= (0.6 - min_lidar) * 4.0

    # If robot has stopped moving forward, small penalty
    if linear < 0.05:
        reward -= 0.02

    return float(reward)


class NavEnv:
    """
    The simulation environment.
    In real deployment this connects to ROS2 topics.
    For now it uses a simplified internal model for testing.
    """

    def __init__(self, lidar_size=24, max_range=3.5):
        self.lidar_size = lidar_size        # 24 LiDAR beams
        self.max_range = max_range          # max detection distance
        self.goal = np.array([4.0, 0.0], dtype=np.float32)
        self.position = np.array([0.0, 0.0], dtype=np.float32)
        self.theta = 0.0                    # robot's current angle
        self.prev_goal_dist = None
        self.step_count = 0
        self.lidar_history = deque(maxlen=4)

    def reset(self):
        """Reset robot to starting position for a new episode."""
        self.position[:] = 0.0
        self.theta = 0.0
        # NOVELTY 3: Random goal position every episode
        self.goal = np.array([
            random.uniform(2.5, 4.5),
            random.uniform(-2.0, 2.0)
        ], dtype=np.float32)
        self.prev_goal_dist = self._goal_distance()
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        """Apply action and return next state, reward, done."""
        linear = float(np.clip(action[0], 0.0, 0.35))   # forward speed
        angular = float(np.clip(action[1], -1.2, 1.2))  # turning speed
        dt = 0.1

        # Move robot
        self.theta += angular * dt
        self.position[0] += linear * math.cos(self.theta) * dt
        self.position[1] += linear * math.sin(self.theta) * dt
        self.step_count += 1

        obs = self._get_obs()
        reward, done, info = self._reward_done_info(linear, angular, obs)
        return obs, reward, done, info

    def _goal_distance(self):
        return float(np.linalg.norm(self.goal - self.position))

    def _goal_angle(self):
        dx, dy = self.goal - self.position
        return float(math.atan2(dy, dx) - self.theta)

    def _fake_lidar(self):
        """
        Fake LiDAR for testing without Gazebo.
        Replace this with real /scan data in ROS2 node.
        """
        base = np.ones(self.lidar_size, dtype=np.float32) * self.max_range
        noise = np.random.normal(0.0, 0.03, size=self.lidar_size).astype(np.float32)
        return np.clip(base + noise, 0.05, self.max_range)

    def _get_obs(self):
        """
        Build the state vector the DRL brain sees:
        [24 LiDAR readings (normalized), goal_distance, goal_angle]
        Total: 26 numbers
        """
        lidar = self._fake_lidar()
        self.lidar_history.append(lidar)
        goal_dist = self._goal_distance()
        goal_ang = self._goal_angle()
        obs = np.concatenate([
            lidar / self.max_range,                                          # normalize 0-1
            np.array([goal_dist / 5.0, goal_ang / math.pi], dtype=np.float32)  # normalize
        ])
        return obs.astype(np.float32)

    def _reward_done_info(self, linear, angular, obs):
        goal_dist = self._goal_distance()
        min_lidar = float(np.min(obs[:-2]) * self.max_range)
        progress = 0.0 if self.prev_goal_dist is None else self.prev_goal_dist - goal_dist
        self.prev_goal_dist = goal_dist

        reward = adaptive_reward(goal_dist, progress, min_lidar,
                                  linear, angular, self.step_count)
        done = False
        info = {"goal_distance": goal_dist, "min_lidar": min_lidar}

        # Collision
        if min_lidar < 0.18:
            reward -= 80.0
            done = True
            info["collision"] = True

        # Success
        if goal_dist < 0.25:
            reward += 120.0
            done = True
            info["success"] = True

        # Timeout
        if self.step_count >= 500:
            done = True
            info["timeout"] = True

        return reward, done, info