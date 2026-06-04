"""Bare-minimum REINFORCE agent for Polytopia.
Trains a small MLP to play via vanilla policy gradient."""
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tribes_env import TribesEnv

# ---------- Featurizer ----------
# Brutally simple: flatten the board into a fixed-size vector.
# We pull a handful of scalar features per tile + tribe scalars.
# No embeddings, no attention — just enough signal for the NN.

NUM_TILES = 121  # 11x11 board
FEATURES_PER_TILE = 4  # terrain_id, unit_present, owner_id, city_present
NUM_TRIBE_SCALARS = 6  # active_tribe, tick, score, stars, num_cities, num_units
FEATURE_DIM = NUM_TILES * FEATURES_PER_TILE + NUM_TRIBE_SCALARS  # 490
MAX_ACTIONS = 8192

def hash_str(s, mod=20):
    """Stable small integer from a string. For unknown vocab values."""
    return abs(hash(s)) % mod

def featurize(state):
    """Turn a state dict into a fixed-length float vector."""
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    board = state.get("board", {})

    # Board can be dict of dicts or nested list — handle both
    tiles = []
    if isinstance(board, dict):
        # Try common shapes
        for k, v in board.items():
            if isinstance(v, list):
                for row in v:
                    if isinstance(row, list):
                        tiles.extend(row)
                    else:
                        tiles.append(row)
                break
        if not tiles:
            tiles = list(board.values())
    elif isinstance(board, list):
        for row in board:
            if isinstance(row, list):
                tiles.extend(row)
            else:
                tiles.append(row)

    for i, tile in enumerate(tiles[:NUM_TILES]):
        if not isinstance(tile, dict):
            continue
        base = i * FEATURES_PER_TILE
        vec[base + 0] = hash_str(str(tile.get("terrain", "")), 20) / 20.0
        vec[base + 1] = 1.0 if tile.get("unit") else 0.0
        vec[base + 2] = (tile.get("tribeId", -1) + 1) / 5.0
        vec[base + 3] = 1.0 if tile.get("city") else 0.0

    # Tribe scalars
    active = state.get("activeTribeID", 0)
    tribes = state.get("tribes", {})
    tribe_data = tribes.get(str(active), {}) if isinstance(tribes, dict) else {}
    base = NUM_TILES * FEATURES_PER_TILE
    vec[base + 0] = active / 5.0
    vec[base + 1] = state.get("tick", 0) / 100.0
    vec[base + 2] = tribe_data.get("score", 0) / 1000.0
    vec[base + 3] = tribe_data.get("stars", 0) / 100.0
    vec[base + 4] = len(tribe_data.get("cities", []) or tribe_data.get("citiesID", [])) / 10.0
    vec[base + 5] = len(tribe_data.get("units", []) or tribe_data.get("unitsID", [])) / 20.0

    return vec

# ---------- Network ----------
class PolicyNet(nn.Module):
    def __init__(self, in_dim=FEATURE_DIM, hidden=256, out_dim=MAX_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, mask):
        logits = self.net(x)
        # Mask illegal actions
        logits = logits.masked_fill(~mask, -1e9)
        return logits

# ---------- Training ----------
def run_episode(env, model, deterministic=False):
    """Play one game. Returns (log_probs, rewards, final_score)."""
    obs, info = env.reset()
    state = json.loads(obs["state_json"])
    log_probs = []
    rewards = []
    total_reward = 0.0
    steps = 0
    MAX_STEPS = 200  # safety cap

    while steps < MAX_STEPS:
        feats = torch.from_numpy(featurize(state)).unsqueeze(0)
        mask_list = info["action_mask"]
        mask = torch.tensor(mask_list, dtype=torch.bool).unsqueeze(0)

        if not mask.any():
            break

        logits = model(feats, mask)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = int(torch.argmax(probs, dim=-1).item())
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action_t = dist.sample()
            action = int(action_t.item())
            log_probs.append(dist.log_prob(action_t))

        obs, reward, term, trunc, info = env.step(action)
        state = json.loads(obs["state_json"])
        rewards.append(reward)
        total_reward += reward
        steps += 1
        if term or trunc:
            break

    return log_probs, rewards, total_reward, steps

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def main():
    print("Initializing environment and model...")
    env = TribesEnv(level_file="tribes/levels/SampleLevel.csv")
    model = PolicyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    NUM_EPISODES = 1000  # adjust based on time
    log = []
    start = time.time()

    print(f"Training for {NUM_EPISODES} episodes...")
    for ep in range(NUM_EPISODES):
        log_probs, rewards, total, steps = run_episode(env, model)

        if not log_probs:
            print(f"Episode {ep}: no legal actions taken, skipping")
            continue

        returns = compute_returns(rewards)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        # Normalize returns for stability
        if returns_t.std() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # REINFORCE loss
        loss = -torch.stack([lp * R for lp, R in zip(log_probs, returns_t)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed = time.time() - start
        log.append({"episode": ep, "reward": total, "steps": steps, "loss": float(loss.item())})
        if ep % 5 == 0:
            recent = log[-20:]
            avg_r = sum(x["reward"] for x in recent) / len(recent)
            print(f"Ep {ep:4d} | reward={total:7.1f} | steps={steps:3d} | "
                  f"avg20={avg_r:7.1f} | loss={loss.item():8.2f} | "
                  f"elapsed={elapsed/60:.1f}m")

        # Save checkpoint every 50 eps
        if ep > 0 and ep % 50 == 0:
            torch.save(model.state_dict(), "policy_checkpoint.pt")
            with open("training_log.json", "w") as f:
                json.dump(log, f, indent=2)

    # Final save
    torch.save(model.state_dict(), "policy_final.pt")
    with open("training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. Saved policy_final.pt and training_log.json")
    print(f"Total time: {(time.time()-start)/60:.1f} minutes")
    env.close()

if __name__ == "__main__":
    main()