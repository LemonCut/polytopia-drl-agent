# Polytopia DRL Agent

Project for CSE 190 (Deep Reinforcement Learning) at UCSD, Spring 2026.

Uses [Tribes](https://github.com/GAIGResearch/Tribes) as an environment and benchmark.

# Notes for team

`play-tribes.py` runs the Java game using the config in `tribes/play.json`.

`tribes_bridge.py` is a Python wrapper around the Java runtime.

`tribes_env.py` is a Gymnasium-compatible environment wrapper for proof-of-concept RL work.

# Gymnasium Wrapper

For debugging and early DRL experiments, the Gymnasium wrapper returns the full Tribes game state as JSON and exposes the current legal actions through a masked discrete action space.

```python
from tribes_env import TribesEnv

env = TribesEnv(level_file="tribes/levels/SampleLevel.csv", game_mode="CAPITALS", seed=123)
obs, info = env.reset()
print(obs["state_json"])
print(info["legal_actions"][0])

obs, reward, terminated, truncated, info = env.step(0)
print(reward, terminated, truncated)
env.close()
```

The action mask is available through `env.action_masks()`, and the raw legal action list is in `info["legal_actions"]`.
