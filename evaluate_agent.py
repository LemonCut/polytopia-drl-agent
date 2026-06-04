"""Evaluate trained agent vs random and report win rates.
Uses stochastic sampling (not argmax) to avoid deterministic collapse at eval time."""
import json
import torch
import numpy as np
from tribes_env import TribesEnv
from train_agent import PolicyNet, featurize, FEATURE_DIM, MAX_ACTIONS

def play_one(env, policy=None):
    """Play one game with the given policy (None = random). Return final score."""
    obs, info = env.reset()
    state = json.loads(obs["state_json"])
    steps = 0
    while steps < 300:
        mask = info["action_mask"]
        legal_idxs = [i for i, m in enumerate(mask) if m]
        if not legal_idxs:
            break
        if policy is None:
            action = int(np.random.choice(legal_idxs))
        else:
            feats = torch.from_numpy(featurize(state)).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
            with torch.no_grad():
                logits = policy(feats, mask_t)
                probs = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, 1).item())
        obs, reward, term, trunc, info = env.step(action)
        state = json.loads(obs["state_json"])
        steps += 1
        if term or trunc:
            break
    # Final score
    active = state.get("activeTribeID", 0)
    score = state.get("tribes", {}).get(str(active), {}).get("score", 0)
    return score, steps

def main():
    env = TribesEnv(level_file="tribes/levels/SampleLevel.csv")

    # Load trained policy (v2 = entropy-regularized version)
    policy = PolicyNet()
    policy.load_state_dict(torch.load("policy_v2.pt"))
    policy.eval()

    N = 10
    print(f"Playing {N} games as random baseline...")
    random_scores = []
    for i in range(N):
        s, steps = play_one(env, policy=None)
        random_scores.append(s)
        print(f"  Random game {i}: score={s}, steps={steps}")

    print(f"\nPlaying {N} games as trained agent (v2, stochastic sampling)...")
    agent_scores = []
    for i in range(N):
        s, steps = play_one(env, policy=policy)
        agent_scores.append(s)
        print(f"  Agent game {i}: score={s}, steps={steps}")

    print(f"\n=== RESULTS ===")
    print(f"Random  avg score: {np.mean(random_scores):.1f} ± {np.std(random_scores):.1f}")
    print(f"Trained avg score: {np.mean(agent_scores):.1f} ± {np.std(agent_scores):.1f}")
    print(f"Improvement: {np.mean(agent_scores) - np.mean(random_scores):+.1f}")

    with open("eval_results_v2.json", "w") as f:
        json.dump({
            "random_scores": random_scores,
            "agent_scores": agent_scores,
            "random_mean": float(np.mean(random_scores)),
            "agent_mean": float(np.mean(agent_scores)),
        }, f, indent=2)

if __name__ == "__main__":
    main()