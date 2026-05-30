"""Capture sample game states to disk for offline featurizer design."""
import json
import os
from tribes_env import TribesEnv

env = TribesEnv(level_file='tribes/levels/SampleLevel.csv')
samples = []

# Capture across a few resets and a few steps each for variety

for game in range(3):
    obs, info = env.reset(seed=game)
    state = json.loads(obs['state_json'])
    samples.append({
        "game": game,
        "step": 0,
        "state": state,
        "legal_actions": info['legal_actions'],
        "num_legal": len(info['legal_actions']),
    })
    print(f"Game {game} step 0: {len(info['legal_actions'])} legal actions")

    # Take up to 5 legal steps to see how state evolves
    for step in range(1, 6):
        if not info['legal_actions']:
            break
        # Pick first legal action (deterministic, for reproducibility)
        obs, reward, term, trunc, info = env.step(0)
        state = json.loads(obs['state_json'])
        samples.append({
            "game": game,
            "step": step,
            "state": state,
            "legal_actions": info['legal_actions'],
            "num_legal": len(info['legal_actions']),
            "reward": reward,
        })
        print(f"Game {game} step {step}: {len(info['legal_actions'])} legal, reward={reward}")
        if term or trunc:
            print("  -> game ended")
            break

env.close()

with open("sample_states.json", "w") as f:
    json.dump(samples, f, indent=2)

print(f"\nSaved {len(samples)} sample states to sample_states.json")
print(f"File size: {os.path.getsize('sample_states.json') / 1024:.1f} KB")