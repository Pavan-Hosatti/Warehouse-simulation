# train.py — Full Training Loop with TD3
# Run this: python3 train.py

from env import NavEnv
from agent_td3 import TD3Agent
from safety import SafetySupervisor
from randomize import apply_domain_randomization, noisy_lidar
from telemetry import Telemetry
import numpy as np

# ── CONFIG ─────────────────────────────────
STATE_DIM    = 26
ACTION_DIM   = 2
MAX_ACTION   = 1.0
EPISODES     = 1000
MAX_STEPS    = 500
BATCH_SIZE   = 256
SAVE_EVERY   = 100
PRINT_EVERY  = 10
# ───────────────────────────────────────────

def train():
    env      = NavEnv()
    agent    = TD3Agent(STATE_DIM, ACTION_DIM, MAX_ACTION)
    safety   = SafetySupervisor()
    telemetry = Telemetry()

    print("=" * 55)
    print("  Schrodinger's Bug — DRL Training Starting")
    print(f"  Episodes: {EPISODES} | State: {STATE_DIM}D | Action: {ACTION_DIM}D")
    print("=" * 55)

    best_success_rate = 0.0

    for ep in range(EPISODES):

        # NOVELTY 3: New random world every episode
        env = apply_domain_randomization(env)
        state = env.reset()

        total_reward = 0.0
        collision    = 0
        success      = 0

        for t in range(MAX_STEPS):

            # NOVELTY 3: Add noise to LiDAR
            lidar_noisy = noisy_lidar(state[:-2])
            state_noisy = np.concatenate([lidar_noisy, state[-2:]])

            # TD3 action
            noise_std = max(0.05, 0.3 - ep * 0.0003)
            action = agent.select_action(state_noisy, noise_std=noise_std)

            # NOVELTY 2: Safety supervisor
            safe_action = safety.filter_action(action, lidar_noisy)

            # Step
            next_state, reward, done, info = env.step(safe_action)
            total_reward += reward

            # Store
            agent.buffer.add(state_noisy, safe_action, reward, next_state, done)

            # Learn
            agent.train_step_update(BATCH_SIZE)

            state = next_state

            if info.get("collision"): collision = 1
            if info.get("success"):   success   = 1
            if done: break

        # NOVELTY 4: Telemetry
        telemetry.log_episode(total_reward, success, collision, t + 1)

        # Print stats
        if (ep + 1) % PRINT_EVERY == 0:
            stats = telemetry.summary()
            print(f"\nEpisode {ep+1:4d}/{EPISODES}")
            print(f"  Success Rate  : {stats['success_rate']:5.1f}%")
            print(f"  Collision Rate: {stats['collision_rate']:5.1f}%")
            print(f"  Mean Reward   : {stats['mean_reward']:7.2f}")
            print(f"  Mean Steps    : {stats['mean_steps']:6.1f}")

        # Save models
        if (ep + 1) % SAVE_EVERY == 0:
            agent.save("models")
            stats = telemetry.summary()
            if stats['success_rate'] > best_success_rate:
                best_success_rate = stats['success_rate']
                agent.save("models/best")
                print(f"  ★ New best model saved! Success: {best_success_rate:.1f}%")

    # ✅ NEW BLOCK — SAVE TRAINING LOG
    np.save("training_log.npy", {
        'rewards':    telemetry.rewards,
        'success':    telemetry.success,
        'collisions': telemetry.collisions,
        'steps':      telemetry.steps,
    })
    print("[SAVED] Training log saved to training_log.npy")

    print("\n" + "=" * 55)
    print("  Training Complete!")
    print(f"  Best Success Rate: {best_success_rate:.1f}%")
    print("=" * 55)

    agent.save("models/final")


if __name__ == "__main__":
    train()