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

## DQN Training

`dqn.py` now provides a masked Deep Q-Network implementation on top of `TribesEnv`.

Train on the sample level:

```bash
python dqn.py train --level-file tribes/levels/SampleLevel.csv --game-mode SCORE --train-episodes 100 --no-resume
```

By default, training starts fresh instead of resuming an old checkpoint, decays exploration faster than the earlier prototype, and scales rewards before they hit the replay buffer for more stable optimization. You can turn on periodic greedy evaluation during training with `--eval-every-episodes` and `--eval-episodes-during-train`, and adjust reward scaling with `--reward-scale`.

Evaluate a saved checkpoint:

```bash
python dqn.py eval --level-file tribes/levels/SampleLevel.csv --game-mode SCORE --checkpoint-path checkpoints/dqn.pt
```

The agent encodes the JSON observation into fixed-size features, respects the action mask when choosing actions, and saves checkpoints with model and optimizer state so training can be resumed.
