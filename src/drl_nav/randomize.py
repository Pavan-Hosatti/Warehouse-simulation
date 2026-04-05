# randomize.py — Domain Randomization
# NOVELTY 3: Every episode the world is slightly different.
# Random start position, random goal, random LiDAR noise.
# This prevents the robot from memorizing one scenario.
# Result: robot works in ANY environment, not just training world.

import random
import numpy as np

def apply_domain_randomization(env):
    """Randomize start and goal every episode."""
    # Slightly random start position
    env.position[:] = [
        random.uniform(-0.2, 0.2),
        random.uniform(-0.2, 0.2)
    ]
    # Random starting angle
    env.theta = random.uniform(-0.4, 0.4)

    # Completely random goal location
    env.goal = np.array([
        random.uniform(2.5, 5.0),
        random.uniform(-2.5, 2.5)
    ], dtype=np.float32)

    return env


def noisy_lidar(lidar, sigma=0.04, dropout=0.02):
    """
    Add realistic noise to LiDAR readings.
    sigma: how much random noise to add
    dropout: chance of a beam returning max range (beam lost)
    """
    lidar = lidar.copy()
    # Random beam dropout (some beams return nothing)
    mask = np.random.rand(*lidar.shape) < dropout
    lidar[mask] = lidar.max()
    # Gaussian noise on all beams
    lidar += np.random.normal(0.0, sigma, size=lidar.shape).astype(lidar.dtype)
    return np.clip(lidar, 0.05, 3.5)