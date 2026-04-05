# Warehouse DRL Navigation — Schrodinger's Bug

This repository contains a Deep Reinforcement Learning (DRL) agent trained to navigate a warehouse environment using ROS2 Humble and Gazebo. It uses the **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** algorithm for smooth, continuous robot control.

## 🚀 One-Click Demo (Local Linux Laptop)

If you have a Linux laptop with ROS2 Humble installed, you can launch the entire simulation — including the Gazebo world, the robot brain, and training metrics — with a single command.

### 1. Prerequisites

Ensure you have the following installed on your host machine:

-   **Ubuntu 22.04** (Recommended)
-   **ROS2 Humble Hawksbill** (Desktop Install)
-   **Gazebo11** (comes with ROS2 Desktop)
-   **Python 3.10+** with these libraries:
    ```bash
    pip3 install torch numpy matplotlib
    ```
-   **Terminal Emulator**: Either `gnome-terminal` (default on Ubuntu) or `xterm`.

### 2. Launch the Simulation

From the root of the repository, navigate to the source folder and run the demo script:

```bash
cd src/drl_nav
bash run_demo.sh
```

> [!NOTE]
> **Wait for the Robot**: The Gazebo world will load first. There is a **15-second delay** programmed into the script to ensure the simulation server is fully ready before the TurtleBot3 is spawned. This prevents common race-condition errors.

### 3. Interactive Commands

The `run_demo.sh` script supports several flags for manual control:

-   **Spawn Obstacles**: `bash run_demo.sh --obstacles 5` (Adds 5 random cylinders)
-   **Clear Obstacles**: `bash run_demo.sh --clear`
-   **Stop Cleanly**: `bash run_demo.sh --stop` (Kills all ROS nodes and Gazebo processes)

---

## 📂 Repository Structure

-   `src/drl_nav/`: Core logic and launch scripts.
    -   `ros2_node.py`: The bridge between the DRL brain and ROS2 topics.
    -   `agent_td3.py`: The TD3 neural network implementation (Actor/Critic).
    -   `models/`: Pre-trained weights for the agent.
    -   `run_demo.sh`: The master control script.
-   `md/`: Documentation and project notes.

## 🤝 Contributing / Forking

1.  **Fork** this repository.
2.  Enable **GitHub Pages** if you want to host the project documentation on a website.
3.  Add your own improvements to the DRL agent or the warehouse environment!

---

**Developed for Schrodinger's Bug — Integrated Warehouse DRL Demo**
