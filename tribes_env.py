from __future__ import annotations

import json
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tribes_bridge import TribesBridge


DEFAULT_MAX_ACTIONS = 8192
DEFAULT_JSON_MAX_LENGTH = 2_000_000


class TribesEnv(gym.Env):
	metadata = {"render_modes": ["human", "json"]}

	def __init__(
		self,
		*,
		level_file: str | None = None,
		level_seed: int | None = None,
		tribes: list[int | str] | None = None,
		game_mode: str | int = "SCORE",
		seed: int | None = None,
		max_episode_steps: int | None = None,
		max_actions: int = DEFAULT_MAX_ACTIONS,
		json_max_length: int = DEFAULT_JSON_MAX_LENGTH,
		render_mode: str | None = None,
		compile_first: bool = True,
	) -> None:
		super().__init__()
		self.level_file = level_file
		self.level_seed = level_seed
		self.tribes = tribes
		self.game_mode = game_mode
		self.seed_value = seed
		self.max_episode_steps = max_episode_steps
		self.max_actions = max_actions
		self.json_max_length = json_max_length
		self.render_mode = render_mode
		self.bridge = TribesBridge(compile_first=compile_first)
		self.action_space = spaces.Discrete(self.max_actions)
		self.observation_space = spaces.Dict(
			{"state_json": spaces.Text(max_length=self.json_max_length)}
		)
		self._state: dict[str, Any] | None = None
		self._legal_actions: list[dict[str, Any]] = []
		self._steps = 0

	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
		super().reset(seed=seed)
		self._steps = 0

		options = options or {}
		level_file = options.get("level_file", self.level_file)
		level_seed = options.get("level_seed", self.level_seed)
		tribes = options.get("tribes", self.tribes)
		agents = options.get("agents", None)
		visuals = options.get("visuals", False)
		game_mode = options.get("game_mode", self.game_mode)
		seed_value = options.get("seed", seed if seed is not None else self.seed_value)

		if level_file is not None:
			response = self.bridge.reset(
				level_file=level_file,
				game_mode=game_mode,
				seed=seed_value,
				agents=agents,
				visuals=visuals,
			)
		elif level_seed is not None and tribes is not None:
			response = self.bridge.reset(
				level_seed=level_seed,
				tribes=tribes,
				game_mode=game_mode,
				seed=seed_value,
				agents=agents,
				visuals=visuals,
			)
		else:
			raise ValueError("TribesEnv.reset requires level_file or level_seed + tribes")

		self._set_state(response, self.bridge.actions())
		observation = self._make_observation(self._state)
		info = self._make_info()
		return observation, info

	def step(self, action: int):
		self._ensure_ready()
		action_index = int(action)
		if action_index < 0 or action_index >= self.max_actions:
			raise ValueError(f"Action {action_index} is outside the action space")

		mask = self.action_masks()
		if not mask[action_index]:
			raise ValueError(f"Action {action_index} is masked out for the current state")

		previous_state = self._state
		response = self.bridge.step(action_index)
		self._steps += 1
		self._set_state(response["state"], response.get("actions"))

		reward = self._compute_reward(previous_state, self._state)
		terminated = bool(self._state.get("gameIsOver", False))
		truncated = bool(self.max_episode_steps is not None and self._steps >= self.max_episode_steps and not terminated)

		observation = self._make_observation(self._state)
		info = self._make_info()
		info["chosen_action_index"] = action_index
		info["chosen_action"] = self._legal_actions[action_index] if action_index < len(self._legal_actions) else None
		return observation, reward, terminated, truncated, info

	def agent_step(self):
		self._ensure_ready()

		previous_state = self._state
		response = self.bridge.agent_step()
		self._steps += 1
		self._set_state(response["state"], response.get("actions"))

		reward = self._compute_reward(previous_state, self._state)
		terminated = bool(self._state.get("gameIsOver", False))
		truncated = bool(self.max_episode_steps is not None and self._steps >= self.max_episode_steps and not terminated)

		observation = self._make_observation(self._state)
		info = self._make_info()
		info["chosen_action_index"] = -1
		info["chosen_action"] = None
		return observation, reward, terminated, truncated, info

	def action_masks(self) -> np.ndarray:
		mask = np.zeros(self.max_actions, dtype=np.bool_)
		mask[: min(len(self._legal_actions), self.max_actions)] = True
		return mask

	def legal_actions(self) -> list[dict[str, Any]]:
		return list(self._legal_actions)

	def render(self):
		self._ensure_ready()
		return self._state if self.render_mode == "json" else self._make_observation(self._state)

	def close(self) -> None:
		self.bridge.close()

	def _set_state(self, state: dict[str, Any], legal_actions: list[dict[str, Any]] | None = None) -> None:
		self._state = state
		self._legal_actions = legal_actions or []

		if len(self._legal_actions) > self.max_actions:
			raise ValueError(
				f"Current legal action count {len(self._legal_actions)} exceeds max_actions={self.max_actions}"
			)

	def _ensure_ready(self) -> None:
		if self._state is None:
			raise RuntimeError("Call reset() before step() or render()")

	def _make_observation(self, state: dict[str, Any] | None) -> dict[str, str]:
		if state is None:
			raise RuntimeError("Environment has not been reset")
		return {"state_json": json.dumps(state, separators=(",", ":"), sort_keys=True)}

	def _make_info(self) -> dict[str, Any]:
		return {
			"state": self._state,
			"legal_actions": self.legal_actions(),
			"action_mask": self.action_masks().tolist(),
			"active_tribe_id": None if self._state is None else self._state.get("activeTribeID"),
			"tick": None if self._state is None else self._state.get("tick"),
		}

	def _compute_reward(self, previous_state: dict[str, Any] | None, current_state: dict[str, Any]) -> float:
		if previous_state is None:
			return 0.0

		active_tribe_id = previous_state.get("activeTribeID")
		if active_tribe_id is None:
			return 0.0

		tribe_key = str(active_tribe_id)
		previous_score = previous_state.get("tribes", {}).get(tribe_key, {}).get("score", 0)
		current_score = current_state.get("tribes", {}).get(tribe_key, {}).get("score", previous_score)
		reward = float(current_score - previous_score)

		if current_state.get("gameIsOver", False):
			winner = current_state.get("tribes", {}).get(tribe_key, {}).get("winner")
			if winner == 0:
				reward += 100.0
			elif winner == 1:
				reward -= 100.0

		return reward
