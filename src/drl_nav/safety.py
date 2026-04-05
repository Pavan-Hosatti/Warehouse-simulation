# safety.py — The Safety Supervisor
# NOVELTY 2: This sits ON TOP of the neural network.
# No matter what the AI decides, if the robot is too close
# to a wall, this OVERRIDES the decision.
# This is what makes real-world deployment safe.

import numpy as np

class SafetySupervisor:
    def __init__(self, emergency_threshold=0.22, slow_threshold=0.45):
        # emergency_threshold: closer than this = STOP immediately
        # slow_threshold: closer than this = slow down
        self.emergency_threshold = emergency_threshold
        self.slow_threshold = slow_threshold

    def filter_action(self, action, lidar):
        """
        Takes the AI's action and either passes it through,
        slows it down, or completely overrides it.
        """
        min_lidar = float(np.min(lidar))
        linear = float(action[0])
        angular = float(action[1])

        # EMERGENCY: Too close to wall — ignore AI completely
        if min_lidar < self.emergency_threshold:
            print(f"[SAFETY] EMERGENCY OVERRIDE — LiDAR: {min_lidar:.2f}m")
            return [0.0, 1.0]  # stop and turn

        # WARNING: Getting close — slow down
        if min_lidar < self.slow_threshold:
            linear = min(linear, 0.08)
            print(f"[SAFETY] Slowing down — LiDAR: {min_lidar:.2f}m")

        return [linear, angular]