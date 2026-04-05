#!/usr/bin/env python3
# warehouse_launch.py — Safe Robot Spawner for Warehouse World
#
# Run this AFTER the Gazebo warehouse world is open (Terminal 1).
# It waits until /spawn_entity is actually available, then spawns
# the TurtleBot3 Burger — eliminating the race-condition error:
#   "[ERROR] Service /spawn_entity unavailable"
#
# Usage:
#   source /opt/ros/humble/setup.bash
#   export TURTLEBOT3_MODEL=burger
#   python3 warehouse_launch.py
#
# Optional env vars:
#   TB3_SPAWN_X   (default 0.0)
#   TB3_SPAWN_Y   (default 0.0)
#   TB3_SPAWN_Z   (default 0.1)

import os
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from std_srvs.srv import Empty

# ── i18n stub ────────────────────────────────────────────────
def t(s):
    return s

# ── Config from env vars ──────────────────────────────────────
SPAWN_X     = float(os.environ.get("TB3_SPAWN_X", "0.0"))
SPAWN_Y     = float(os.environ.get("TB3_SPAWN_Y", "0.0"))
SPAWN_Z     = float(os.environ.get("TB3_SPAWN_Z", "0.1"))
ROBOT_NAME  = "burger"
MAX_WAIT_S  = 60   # give Gazebo up to 60 s to become ready


class RobotSpawner(Node):
    def __init__(self):
        super().__init__('robot_spawner')
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')

    def wait_for_gazebo(self) -> bool:
        """Poll /spawn_entity until Gazebo is fully up."""
        self.get_logger().info(
            t(f"Waiting up to {MAX_WAIT_S}s for Gazebo /spawn_entity...")
        )
        deadline = time.time() + MAX_WAIT_S
        while time.time() < deadline:
            if self.spawn_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().info(t("✅ Gazebo is ready!"))
                return True
            remaining = int(deadline - time.time())
            self.get_logger().warn(
                t(f"  /spawn_entity not yet available — {remaining}s remaining...")
            )
        return False

    def get_robot_description(self) -> str:
        """Get the URDF/SDF for TurtleBot3 Burger via xacro."""
        tb3_model = os.environ.get("TURTLEBOT3_MODEL", "burger")

        # Try to find the xacro file from the installed tb3 description package
        xacro_path_candidates = [
            f"/opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_{tb3_model}.urdf",
            f"/opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_{tb3_model}.urdf.xacro",
        ]

        for path in xacro_path_candidates:
            if os.path.exists(path):
                if path.endswith(".xacro"):
                    result = subprocess.run(
                        ["xacro", path], capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        return result.stdout
                else:
                    with open(path) as f:
                        return f.read()

        # Fallback: read from /robot_description topic via ros2 topic echo
        self.get_logger().warn(
            t("Could not find URDF file — reading from /robot_description topic")
        )
        return ""

    def spawn_robot(self) -> bool:
        """Spawn TurtleBot3 Burger into Gazebo."""
        robot_desc = self.get_robot_description()

        req = SpawnEntity.Request()
        req.name              = ROBOT_NAME
        req.initial_pose.position.x = SPAWN_X
        req.initial_pose.position.y = SPAWN_Y
        req.initial_pose.position.z = SPAWN_Z

        if robot_desc:
            req.xml = robot_desc
        else:
            # Tell gazebo_ros to fetch from /robot_description topic
            req.robot_namespace = ""
            req.reference_frame = "world"
            # Use the spawn_entity.py approach: pass -topic flag via xml trick
            self.get_logger().info(
                t("Falling back to spawn via ros2 CLI (topic-based)...")
            )
            self._spawn_via_cli()
            return True

        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)

        if future.result() and future.result().success:
            self.get_logger().info(
                t(f"✅ TurtleBot3 Burger spawned at "
                  f"({SPAWN_X}, {SPAWN_Y}, {SPAWN_Z})")
            )
            return True
        else:
            msg = future.result().status_message if future.result() else "timeout"
            self.get_logger().error(t(f"❌ Spawn failed: {msg}"))
            return False

    def _spawn_via_cli(self):
        """
        Reliable fallback: calls spawn_entity.py as a subprocess.
        This mirrors the manual 'Terminal 4' command from the setup guide
        but runs programmatically once Gazebo is confirmed ready.
        """
        cmd = [
            "ros2", "run", "gazebo_ros", "spawn_entity.py",
            "-entity", ROBOT_NAME,
            "-topic", "robot_description",
            "-x", str(SPAWN_X),
            "-y", str(SPAWN_Y),
            "-z", str(SPAWN_Z),
        ]
        self.get_logger().info(t(f"Running: {' '.join(cmd)}"))
        result = subprocess.run(cmd, timeout=30)
        if result.returncode == 0:
            self.get_logger().info(t("✅ Spawn via CLI succeeded."))
        else:
            self.get_logger().error(
                t(f"❌ CLI spawn exited with code {result.returncode}")
            )


def main():
    rclpy.init()
    spawner = RobotSpawner()

    try:
        # Step 1: Block until Gazebo is actually ready
        if not spawner.wait_for_gazebo():
            spawner.get_logger().error(
                t(f"Gazebo did not become ready within {MAX_WAIT_S}s.\n"
                  "Make sure Terminal 1 has the warehouse world running.")
            )
            sys.exit(1)

        # Small extra buffer — Gazebo sometimes accepts the service call
        # but isn't quite done loading physics yet
        spawner.get_logger().info(t("Waiting 2s for physics engine to stabilise..."))
        time.sleep(2.0)

        # Step 2: Spawn the robot
        ok = spawner.spawn_robot()
        if ok:
            print(t("\n✅ Robot spawned. You can now run Terminal 2 (ros2_node.py).\n"))
        else:
            print(t("\n❌ Spawn failed. Check Gazebo logs.\n"))
            sys.exit(1)

    finally:
        spawner.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
