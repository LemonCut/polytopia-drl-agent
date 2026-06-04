# %% [markdown]
# # PPO notebook to script conversion

# %% [markdown]
# #### Imports

# %%
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import time
import gc
from copy import deepcopy


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tribes_env import TribesEnv

# Reduce Reuse Recycle!
from dqn import (
    JsonStateEncoder,
    DQNConfig,  # optional for shared defaults? prob not used
)

# %% [markdown]
# #### Config

# %%
@dataclass
class PPOConfig:
    # Environment
    level_file: str = "tribes/levels/SampleLevel.csv"
    game_mode: str = "SCORE"
    seed: int = 123
    compile_first: bool = True

    # State encoder
    encoder_mode: str = "combined"
    engineered_dim: int = 1024
    raw_dim: int = 256

    # Network
    hidden_dim: int = 1024
    residual_blocks: int = 3

    # PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_range: float = 0.2

    entropy_coef: float = 0.01
    value_coef: float = 0.5

    ppo_epochs: int = 4
    minibatch_size: int = 256 # unused as of now, probably a big improvement

    gradient_clip_norm: float = 1.0

    # Rollout collection
    rollout_steps: int = 512 # ?2048
    reward_scale: float = 100.0

    # Training schedule
    total_iterations: int = 1000

    # Evaluation
    eval_every_iterations: int = 25
    eval_episodes: int = 5
    eval_seed_offset: int = 10000
    eval_max_steps_per_episode: int = 512 # ?2048

    # Episode limits
    max_steps_per_episode: int = 2000

    # Checkpoints
    checkpoint_path: str = "checkpoints/ppo.pt"
    resume: bool = True

    # Device
    device: str = "mps" # cpu

# %% [markdown]
# #### Env Helper

# %%
def make_env(config: PPOConfig) -> TribesEnv:
    return TribesEnv(
        level_file=config.level_file,
        game_mode=config.game_mode,
        seed=config.seed,
        compile_first=config.compile_first,
    )


def evaluate_agent(
    agent,
    config,
    encoder,
    episodes,
    seed_offset=0,
):
    env = make_env(config)

    rewards = []

    t0 = time.time()

    print(f"[eval] start episodes={episodes}")

    for episode in range(episodes):

        ep_t0 = time.time()

        obs, info = env.reset(
            seed=config.seed + seed_offset + episode
        )

        state = encoder.encode(obs["state_json"])
        episode_reward = 0.0
        steps = 0

        for _ in range(config.eval_max_steps_per_episode):

            mask = np.asarray(info["action_mask"], dtype=np.bool_)

            action = agent.select_greedy_action(state, mask)

            next_obs, reward, terminated, truncated, next_info = env.step(action)

            episode_reward += reward
            steps += 1

            state = encoder.encode(next_obs["state_json"])
            info = next_info

            if terminated or truncated:
                break

        rewards.append(float(episode_reward))

        ep_time = time.time() - ep_t0

        print(
            f"[eval] episode={episode+1}/{episodes} "
            f"reward={episode_reward:.2f} "
            f"steps={steps} "
            f"time={ep_time:.2f}s"
        )

    env.close()

    mean_reward = float(np.mean(rewards))
    total_time = time.time() - t0

    print(
        f"[eval] done mean_reward={mean_reward:.3f} "
        f"total_time={total_time:.2f}s "
        f"avg_ep_time={total_time/episodes:.2f}s"
    )

    return {
        "episode_rewards": rewards,
        "mean_reward": mean_reward,
    }

# %% [markdown]
# #### Model definition

# %%
class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)

        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x

        x = self.fc1(x)
        x = self.norm1(x)
        x = F.gelu(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = x + residual

        return F.gelu(x)


class PPOModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        residual_blocks: int,
    ):
        super().__init__()

        layers = [
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]

        for _ in range(residual_blocks):
            layers.append(
                ResidualBlock(hidden_dim)
            )

        self.trunk = nn.Sequential(*layers)

        self.policy_head = nn.Linear(
            hidden_dim,
            action_dim,
        )

        self.value_head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(self, states):
        features = self.trunk(states)

        logits = self.policy_head(features)

        values = self.value_head(
            features
        ).squeeze(-1)

        return logits, values

# %% [markdown]
# #### Rollout utilities

# %%
def masked_categorical(logits, action_masks,):
    logits = logits.masked_fill( ~action_masks, torch.finfo(logits.dtype).min,)
    return torch.distributions.Categorical(logits=logits)


def compute_gae(rewards, dones, values, last_value, gamma, gae_lambda,):
    advantages = []

    gae = 0.0

    values = values + [last_value]

    for t in reversed(
        range(len(rewards))
    ):
        delta = (
            rewards[t]
            + gamma
            * values[t + 1]
            * (1.0 - dones[t])
            - values[t]
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * (1.0 - dones[t])
            * gae
        )

        advantages.insert(0, gae)

    returns = [
        adv + value
        for adv, value
        in zip(
            advantages,
            values[:-1],
        )
    ]

    return advantages, returns

@dataclass
class RolloutBuffer:
    states: list
    actions: list
    rewards: list
    dones: list

    values: list
    log_probs: list

    masks: list

    final_state: np.ndarray
    final_mask: np.ndarray

# %% [markdown]
# #### PPO Agent Definition

# %%
class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        config,
    ):
        self.config = config

        self.device = torch.device(
            config.device
        )

        self.model = PPOModel(
            state_dim,
            action_dim,
            config.hidden_dim,
            config.residual_blocks,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )

    @classmethod
    def load(cls, checkpoint_path, config=None):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        if config is None:
            config = PPOConfig(**checkpoint["config"])

        state_dict = checkpoint["model_state_dict"]

        first_weight = state_dict["trunk.0.weight"]

        state_dim = first_weight.shape[1]

        policy_weight = state_dict["policy_head.weight"]

        action_dim = policy_weight.shape[0]

        agent = cls(state_dim,action_dim,config)

        agent.model.load_state_dict(state_dict)

        if ("optimizer_state_dict" in checkpoint):
            try:
                agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass

        agent.model.eval()

        extra = {
            "iteration": checkpoint.get("iteration", 0),
            "best_eval_reward": checkpoint.get("best_eval_reward", None),
        }

        return agent, extra
    
    def select_action( self, state, action_mask, epsilon=0.0, greedy=False):
        if greedy: return self.select_greedy_action(state, action_mask)
        action, _, _ = self.select_action_sample(state, action_mask)
        return action

    @torch.no_grad()
    def select_action_sample(
        self,
        state,
        action_mask,
    ):
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        mask_tensor = torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        logits, values = self.model(
            state_tensor
        )

        dist = masked_categorical(
            logits,
            mask_tensor,
        )

        action = dist.sample()

        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(values.item()),
        )
    
    @torch.no_grad()
    def select_greedy_action(
        self,
        state,
        action_mask,
    ):
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        mask_tensor = torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        logits, _ = self.model(state_tensor)

        logits = logits.masked_fill(
            ~mask_tensor,
            torch.finfo(logits.dtype).min,
        )

        action = torch.argmax(
            logits,
            dim=1,
        )

        return int(action.item())
    
    @torch.no_grad()
    def predict_value(self,state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device,).unsqueeze(0)

        _, value = self.model(state_tensor)

        return float(value.squeeze(0).item())
    
    def update(
        self,
        states,
        actions,
        old_log_probs,
        returns,
        advantages,
        action_masks,
    ):
        logits, values = self.model(
            states
        )

        dist = masked_categorical(
            logits,
            action_masks,
        )

        new_log_probs = dist.log_prob(
            actions
        )

        entropy = dist.entropy()

        ratio = torch.exp(
            new_log_probs
            - old_log_probs
        )

        surr1 = ratio * advantages

        surr2 = (
            torch.clamp(
                ratio,
                1.0 - self.config.clip_range,
                1.0 + self.config.clip_range,
            )
            * advantages
        )

        policy_loss = (
            -torch.min(
                surr1,
                surr2,
            ).mean()
        )

        value_loss = F.mse_loss(
            values,
            returns,
        )

        entropy_loss = entropy.mean()

        loss = (
            policy_loss
            + self.config.value_coef
            * value_loss
            - self.config.entropy_coef
            * entropy_loss
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip_norm,
        )

        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(
                policy_loss.item()
            ),
            "value_loss": float(
                value_loss.item()
            ),
            "entropy": float(
                entropy_loss.item()
            ),
        }

# %% [markdown]
# #### Rollout Collection

# %%
def collect_rollout(
    env,
    agent,
    encoder,
    config,
):
    obs, info = env.reset()

    state = encoder.encode(
        obs["state_json"]
    )

    rollout = RolloutBuffer(
        states=[],
        actions=[],
        rewards=[],
        dones=[],
        values=[],
        log_probs=[],
        masks=[],
        final_state = None,
        final_mask = None
    )

    while (len(rollout.states) < config.rollout_steps):
        mask = np.asarray(
            info["action_mask"],
            dtype=np.bool_,
        )

        (action, log_prob, value,) = agent.select_action_sample(state,mask,)

        (next_obs,reward,terminated,truncated,next_info,) = env.step(action)

        rollout.states.append(state)
        rollout.actions.append(action)
        rollout.rewards.append(reward / config.reward_scale)
        rollout.dones.append(terminated or truncated)
        rollout.values.append(value)
        rollout.log_probs.append(log_prob)
        rollout.masks.append(mask)
        
        state = encoder.encode(next_obs["state_json"])

        info = next_info

        if terminated or truncated:
            obs, info = env.reset()
            state = encoder.encode(obs["state_json"])
    
    rollout.final_state = state
    rollout.final_mask = np.asarray(
        info["action_mask"],
        dtype=np.bool_,
    )

    return rollout