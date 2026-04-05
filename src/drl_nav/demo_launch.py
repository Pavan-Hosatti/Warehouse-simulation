#!/usr/bin/env python3
# demo_launch.py — Fixed Gazebo + TurtleBot3 Launch
#
# WHY THIS EXISTS:
#   turtlebot3_gazebo empty_world.launch.py fires spawn_entity.py
#   immediately after gzserver starts — before the ROS factory plugin
#   (libgazebo_ros_factory.so) has finished loading.
#   Result: "[ERROR] Service /spawn_entity unavailable."
#
#   This file fixes it by using TimerAction to delay the spawn 15s,
#   giving gzserver guaranteed time to load all plugins.
#
# RUN WITH:
#   export TURTLEBOT3_MODEL=burger
#   ros2 launch demo_launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    # ── Package dirs ────────────────────────────────────────────
    pkg_gazebo_ros  = get_package_share_directory('gazebo_ros')
    pkg_tb3_gazebo  = get_package_share_directory('turtlebot3_gazebo')

    tb3_model = os.environ.get('TURTLEBOT3_MODEL', 'burger')
    world_file = os.path.join(pkg_tb3_gazebo, 'worlds', 'turtlebot3_dqn_stage4.world')

    # ── Env fixes ───────────────────────────────────────────────
    # Suppress ALSA "no sound card" spam — harmless on WSL / server
    env_fixes = [
        SetEnvironmentVariable('ALSA_CARD',        'none'),
        SetEnvironmentVariable('SDL_AUDIODRIVER',   'dummy'),
        SetEnvironmentVariable('AUDIODEV',          'null'),
        SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1'),  # software GL fallback
        SetEnvironmentVariable('SVGA_VGPU10',       '0'),
    ]

    # ── 1. gzserver (loads the world + ROS factory plugin) ──────
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_file, 'verbose': 'true'}.items(),
    )

    # ── 2. gzclient (the Gazebo GUI) ────────────────────────────
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        ),
        launch_arguments={'verbose': 'false'}.items(),
    )

    # ── 3. Robot State Publisher ─────────────────────────────────
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_gazebo, 'launch', 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    # ── 4. Spawn robot — DELAYED 15s so gzserver factory plugin ─
    #       is guaranteed to be loaded before we call spawn_entity
    #       MUST use -file with the model.sdf so plugins (LiDAR, Motors) are loaded!
    tb3_sdf_path = os.path.join(
        pkg_tb3_gazebo, 'models', f'turtlebot3_{tb3_model}', 'model.sdf'
    )

    spawn_robot = TimerAction(
        period=15.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                    '-entity', tb3_model,
                    '-file',   tb3_sdf_path,
                    '-x', '-0.5',
                    '-y', '0.5',
                    '-z', '0.01',
                ],
                output='screen',
            )
        ],
    )

    return LaunchDescription(
        env_fixes + [
            gzserver,
            gzclient,
            robot_state_publisher,
            spawn_robot,       # fires 15s after gzserver starts
        ]
    )
