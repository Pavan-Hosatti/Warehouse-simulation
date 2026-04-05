# agent_td3.py — The TD3 Brain
# TD3 = Twin Delayed Deep Deterministic Policy Gradient
#
# WHY TD3 over DQN and DDPG:
# DQN   → only discrete actions (turn left OR right) — jerky robot
# DDPG  → continuous but overestimates Q values — unstable training
# TD3   → fixes DDPG with 3 tricks:
#          1. Twin critics (two judges, take pessimistic score)
#          2. Delayed actor updates (brain updates less often than judges)
#          3. Target policy smoothing (adds noise to prevent overfitting)
#
# INPUT:  26 numbers (24 LiDAR + goal_distance + goal_angle)
# OUTPUT: 2 numbers (linear_velocity, angular_velocity)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# ─────────────────────────────────────────────
# ACTOR NETWORK — The Decision Maker
# Takes state → outputs action
# ─────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, action_dim),nn.Tanh()   # output between -1 and 1
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# CRITIC NETWORK — The Judge
# Takes state + action → outputs a score (Q value)
# TD3 uses TWO critics — takes the lower score
# This prevents overconfidence
# ─────────────────────────────────────────────
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Critic 1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),                    nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Critic 2 (the twin)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),                    nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


# ─────────────────────────────────────────────
# REPLAY BUFFER — Memory Bank
# Stores past experiences so the agent can
# learn from them later (not just latest step)
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100_000):
        self.max_size = max_size
        self.ptr = 0        # current write position
        self.size = 0       # how full the buffer is

        self.states      = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((max_size, 1),          dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((max_size, 1),          dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Store one transition."""
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=256):
        """Randomly sample a batch of past experiences."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor(self.actions[idx]),
            torch.FloatTensor(self.rewards[idx]),
            torch.FloatTensor(self.next_states[idx]),
            torch.FloatTensor(self.dones[idx]),
        )

    def ready(self, batch_size=256):
        """Only start training when buffer has enough data."""
        return self.size >= batch_size


# ─────────────────────────────────────────────
# TD3 AGENT — Puts it all together
# ─────────────────────────────────────────────
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action=1.0, device="cpu"):
        self.device     = device
        self.max_action = max_action
        self.action_dim = action_dim

        # Networks
        self.actor          = Actor(state_dim, action_dim).to(device)
        self.actor_target   = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic         = Critic(state_dim, action_dim).to(device)
        self.critic_target  = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)

        # Replay buffer
        self.buffer = ReplayBuffer(state_dim, action_dim)

        # TD3 hyperparameters
        self.gamma        = 0.99    # discount factor
        self.tau          = 0.005   # soft update rate
        self.policy_noise = 0.2     # noise added to target actions
        self.noise_clip   = 0.5     # max noise magnitude
        self.policy_delay = 2       # actor updates every N critic updates
        self.train_step   = 0       # internal counter

    def select_action(self, state, noise_std=0.1):
        """
        Choose action for current state.
        noise_std > 0 during training (exploration)
        noise_std = 0 during evaluation (pure policy)
        """
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        a = self.actor(s).detach().cpu().numpy()[0]
        if noise_std > 0:
            a += np.random.normal(0, noise_std, size=a.shape)
        a = np.clip(a, -1.0, 1.0)
        return a * self.max_action

    def train_step_update(self, batch_size=256):
        """
        One training update — called every step once buffer is ready.
        This is where the actual learning happens.
        """
        if not self.buffer.ready(batch_size):
            return  # not enough data yet

        self.train_step += 1
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        # ── Critic Update ──────────────────────────────
        with torch.no_grad():
            # Add smoothing noise to target actions (TD3 trick 3)
            noise = (torch.randn_like(actions) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise
                            ).clamp(-self.max_action, self.max_action)

            # Use the LOWER of twin Q values (TD3 trick 1)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * torch.min(q1_target, q2_target)

        q1, q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor Update — Delayed (TD3 trick 2) ───────
        if self.train_step % self.policy_delay == 0:
            actor_loss = -self.critic(states, self.actor(states))[0].mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Soft update both target networks
            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, path="models"):
        """Save trained model to disk."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),  f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")
        print(f"[SAVE] Model saved to {path}/")

    def load(self, path="models"):
        """Load a previously trained model."""
        self.actor.load_state_dict(
            torch.load(f"{path}/actor.pth", map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f"{path}/critic.pth", map_location=self.device)
        )
        print(f"[LOAD] Model loaded from {path}/")