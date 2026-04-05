#!/usr/bin/env python3
# spawn_cylinder.py — Live Obstacle Stress Tester
#
# Drops cylinders at random locations in the Gazebo warehouse world
# to stress-test the DRL brain's obstacle avoidance at run-time.
#
# Usage (while Gazebo is running):
#   source /opt/ros/humble/setup.bash
#   python3 spawn_cylinder.py                  # one cylinder, random pos
#   python3 spawn_cylinder.py --count 5        # five cylinders
#   python3 spawn_cylinder.py --x 1.5 --y 0.5 # specific location
#   python3 spawn_cylinder.py --clear          # remove all test cylinders
#
# Compatible with: ROS2 Humble + Gazebo Classic (gazebo_ros)

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
import random
import argparse
import sys
import time

# ── i18n stub ────────────────────────────────────────────────
def t(s):
    """Wrap all visible UI strings — swap in a real i18n backend here."""
    return s

# ── Hexagon safe spawn zones ────────────────────────────────
# Area inside the stage 4 hexagon world, avoiding the walls.
SPAWN_ZONES = [
    # (x_min, x_max, y_min, y_max)
    (-0.5, 0.5, -0.5, 0.5),   # central open area
    ( 0.0, 0.8, -0.2, 0.2),   # east opening
]

CYLINDER_HEIGHT  = 0.5   # metres
CYLINDER_RADIUS  = 0.15  # metres — roughly a person's leg width
SPAWN_Z          = 0.25  # spawn slightly above floor

# SDF template for a simple coloured cylinder
CYLINDER_SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{height}</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{height}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.1</ixx><iyy>0.1</iyy><izz>0.05</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
    </link>
  </model>
</sdf>"""


class CylinderSpawner(Node):
    def __init__(self):
        super().__init__('cylinder_spawner')
        self._spawned_names: list[str] = []

        self.spawn_client  = self.create_client(SpawnEntity,  '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')

    # ── Internal helpers ──────────────────────────────────────

    def _wait_for_service(self, client, name: str, timeout: float = 10.0):
        self.get_logger().info(t(f"Waiting for {name} service..."))
        ok = client.wait_for_service(timeout_sec=timeout)
        if not ok:
            self.get_logger().error(
                t(f"Service {name} unavailable after {timeout}s. "
                  "Is Gazebo running?")
            )
            return False
        self.get_logger().info(t(f"{name} service ready."))
        return True

    def _random_pose(self):
        zone = random.choice(SPAWN_ZONES)
        x = random.uniform(zone[0], zone[1])
        y = random.uniform(zone[2], zone[3])
        return x, y

    def _pick_colour(self):
        """Bright traffic-cone orange by default; randomise for multiple."""
        palettes = [
            (1.0, 0.4, 0.0),   # orange  — person / cone
            (1.0, 0.1, 0.1),   # red     — fire hazard marker
            (0.2, 0.8, 0.2),   # green   — pallet post
            (0.9, 0.9, 0.0),   # yellow  — caution marker
        ]
        return random.choice(palettes)

    # ── Public API ────────────────────────────────────────────

    def spawn_cylinder(self, x: float, y: float, name: str = None) -> bool:
        """Spawn one cylinder at (x, y).  Returns True on success."""
        if not self._wait_for_service(self.spawn_client, '/spawn_entity'):
            return False

        if name is None:
            name = f"drl_obstacle_{int(time.time()*1000) % 100000}"

        r, g, b = self._pick_colour()
        sdf = CYLINDER_SDF_TEMPLATE.format(
            name=name,
            radius=CYLINDER_RADIUS,
            height=CYLINDER_HEIGHT,
            r=r, g=g, b=b,
        )

        req = SpawnEntity.Request()
        req.name            = name
        req.xml             = sdf
        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y)
        req.initial_pose.position.z = SPAWN_Z

        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() and future.result().success:
            self._spawned_names.append(name)
            self.get_logger().info(
                t(f"✅ Spawned '{name}' at ({x:.2f}, {y:.2f})")
            )
            return True
        else:
            msg = future.result().status_message if future.result() else "timeout"
            self.get_logger().error(t(f"❌ Spawn failed: {msg}"))
            return False

    def clear_all(self) -> int:
        """Delete every cylinder spawned this session. Returns count removed."""
        if not self._wait_for_service(self.delete_client, '/delete_entity'):
            return 0

        removed = 0
        for name in list(self._spawned_names):
            req = DeleteEntity.Request()
            req.name = name
            future = self.delete_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            if future.result() and future.result().success:
                self.get_logger().info(t(f"🗑  Removed '{name}'"))
                removed += 1
            else:
                self.get_logger().warn(t(f"Could not remove '{name}'"))
        self._spawned_names.clear()
        return removed


# ── CLI entry point ───────────────────────────────────────────

def parse_args(argv):
    p = argparse.ArgumentParser(
        description=t("Drop cylinders into Gazebo to stress-test DRL obstacle avoidance.")
    )
    p.add_argument("--count", type=int, default=1,
                   help=t("Number of cylinders to spawn (default: 1)"))
    p.add_argument("--x", type=float, default=None,
                   help=t("Fixed X position (overrides random)"))
    p.add_argument("--y", type=float, default=None,
                   help=t("Fixed Y position (overrides random)"))
    p.add_argument("--clear", action="store_true",
                   help=t("Remove all previously spawned test cylinders"))
    p.add_argument("--delay", type=float, default=0.5,
                   help=t("Seconds between spawns when --count > 1 (default: 0.5)"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    rclpy.init()
    spawner = CylinderSpawner()

    try:
        if args.clear:
            n = spawner.clear_all()
            print(t(f"Cleared {n} obstacle(s)."))
            return

        for i in range(args.count):
            if args.x is not None and args.y is not None:
                x, y = args.x, args.y
            else:
                x, y = spawner._random_pose()

            label = f"drl_obstacle_{i+1}"
            ok = spawner.spawn_cylinder(x, y, name=label)
            if not ok:
                print(t(f"[WARN] Failed to spawn cylinder {i+1}/{args.count}"))

            if i < args.count - 1:
                time.sleep(args.delay)

        total = len(spawner._spawned_names)
        print(t(f"\n🚧  {total} obstacle(s) active. Watch the DRL brain react!"))
        print(t("    Run with --clear to remove them all.\n"))

    finally:
        spawner.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
