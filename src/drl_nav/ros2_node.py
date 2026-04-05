# ros2_node.py — Connects DRL brain to real ROS2 topics
#
# Subscribes to:
#   /scan  → real LiDAR readings from TurtleBot3
#   /odom  → real position and velocity
#
# Publishes to:
#   /cmd_vel → actual movement commands to the robot
#
# WAREHOUSE UPGRADE:
#   - Multi-goal waypoint resetting (Goal A → Goal B → Goal A ...)
#   - Safe robot spawn with Gazebo readiness wait
#   - Live Min LiDAR telemetry for Safety Supervisor visibility

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import math
import sys
import os

# Add our drl_nav folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_td3 import TD3Agent
from safety import SafetySupervisor
from telemetry import Telemetry

from rclpy.qos import qos_profile_sensor_data

# ── i18n stub ────────────────────────────────────────────────
def t(s):
    """Wrap all visible UI strings — swap in a real i18n backend here."""
    return s

# ── CONFIG ───────────────────────────────────────────────────
STATE_DIM   = 26
ACTION_DIM  = 2
LIDAR_SIZE  = 24
MAX_RANGE   = 3.5

# Hexagon world waypoint sequence:
# The stage 4 world is a small hexagon.
# We cycle between two safe clearings.
WAYPOINTS = [
    [1.0,  0.5],   # Goal A
    [-1.0, -0.5],  # Goal B
]
GOAL_REACHED_DIST = 0.35   # metres — distance threshold for "arrived"
# ─────────────────────────────────────────────────────────────


class DRLNavigationNode(Node):
    def __init__(self):
        super().__init__('drl_navigation_node')
        self.get_logger().info(t("Schrodinger's Bug — DRL Hexagon Node Starting..."))

        # Load trained agent
        self.agent     = TD3Agent(STATE_DIM, ACTION_DIM)
        self.safety    = SafetySupervisor()
        self.telemetry = Telemetry()

        # Try loading saved model
        try:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/best")
            self.agent.load(model_dir)
            self.get_logger().info(t(f"Loaded trained model from {model_dir}"))
        except Exception:
            self.get_logger().warn(t("No saved model found — running with random policy"))

        # Robot state
        self.lidar_data = np.ones(LIDAR_SIZE, dtype=np.float32) * MAX_RANGE
        self.position   = [0.0, 0.0]
        self.theta      = 0.0

        # ── Multi-goal waypoint state ────────────────────────
        self.waypoint_index = 0                          # which waypoint we're chasing
        self.goal           = list(WAYPOINTS[0])         # current active goal
        self.goals_reached  = 0                          # total waypoints completed
        self.episode_steps  = 0                          # steps in current leg
        # ─────────────────────────────────────────────────────

        # ROS2 subscribers with SensorDataQoS! (BEST_EFFORT / VOLATILE)
        # Gazebo publishes sensors with SensorDataQoS, default 10 is Reliable and will fail silently.
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile_sensor_data
        )

        # ROS2 publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control loop — runs every 100 ms
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            t(f"Node ready — navigating to Goal A: {WAYPOINTS[0]}")
        )

    # ── Sensor callbacks ──────────────────────────────────────

    def scan_callback(self, msg):
        """Receive real LiDAR data from Gazebo."""
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, MAX_RANGE)
        ranges = np.clip(ranges, 0.05, MAX_RANGE)
        indices = np.linspace(0, len(ranges) - 1, LIDAR_SIZE, dtype=int)
        self.lidar_data = ranges[indices]

    def odom_callback(self, msg):
        """Receive real position data from Gazebo."""
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny, cosy)

    # ── State builder ─────────────────────────────────────────

    def get_state(self):
        """Build state vector from real sensor data."""
        dx = self.goal[0] - self.position[0]
        dy = self.goal[1] - self.position[1]
        goal_dist  = math.sqrt(dx**2 + dy**2)
        goal_angle = math.atan2(dy, dx) - self.theta

        state = np.concatenate([
            self.lidar_data / MAX_RANGE,
            np.array([goal_dist / 5.0,
                      goal_angle / math.pi], dtype=np.float32)
        ])
        return state.astype(np.float32)

    # ── Goal management ───────────────────────────────────────

    def _dist_to_goal(self):
        dx = self.goal[0] - self.position[0]
        dy = self.goal[1] - self.position[1]
        return math.sqrt(dx**2 + dy**2)

    def _advance_waypoint(self):
        """Cycle to the next waypoint in the sequence."""
        self.goals_reached  += 1
        self.episode_steps   = 0
        prev_label = self._waypoint_label(self.waypoint_index)

        self.waypoint_index  = (self.waypoint_index + 1) % len(WAYPOINTS)
        self.goal            = list(WAYPOINTS[self.waypoint_index])
        next_label           = self._waypoint_label(self.waypoint_index)

        self.get_logger().info(
            t(f"✅ {prev_label} REACHED! "
              f"(total waypoints: {self.goals_reached}) "
              f"→ Now heading to {next_label}: {self.goal}")
        )

    def _waypoint_label(self, idx):
        labels = ["Goal A (Shelf)", "Goal B (Dock)"]
        return labels[idx] if idx < len(labels) else f"Goal {idx}"

    # ── Main control loop ─────────────────────────────────────

    def control_loop(self):
        """Main loop — runs every 100 ms."""
        self.episode_steps += 1
        state = self.get_state()

        # Get action from TD3 brain (no exploration noise during deploy)
        action = self.agent.select_action(state, noise_std=0.0)

        # NOVELTY 2: Safety supervisor override
        safe_action = self.safety.filter_action(action, self.lidar_data)

        dist = self._dist_to_goal()

        # ── Waypoint reached? → advance to next goal ─────────
        if dist < GOAL_REACHED_DIST:
            self.stop_robot()
            self._advance_waypoint()
            return
        # ─────────────────────────────────────────────────────

        # Publish movement command
        cmd = Twist()
        cmd.linear.x  = float(np.clip(safe_action[0], 0.0, 0.25))
        cmd.angular.z = float(np.clip(safe_action[1], -1.0, 1.0))
        self.cmd_pub.publish(cmd)

        current_label = self._waypoint_label(self.waypoint_index)
        self.get_logger().info(
            t(f"Pos: ({self.position[0]:.2f}, {self.position[1]:.2f}) | "
              f"{current_label} dist: {dist:.2f}m | "
              f"Action: lin={cmd.linear.x:.2f} ang={cmd.angular.z:.2f} | "
              f"Min LiDAR: {np.min(self.lidar_data):.2f}m | "
              f"Waypoints done: {self.goals_reached}")
        )

    def stop_robot(self):
        """Send zero velocity — stop the robot."""
        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = DRLNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(t("Shutting down..."))
        node.stop_robot()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()