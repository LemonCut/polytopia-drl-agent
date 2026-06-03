from __future__ import annotations

import argparse
import json
import math
import random
import logging
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
	import torch
	from torch import nn
except Exception as exc:  # pragma: no cover - imported at runtime
	raise RuntimeError(
		"dqn.py requires PyTorch. Install torch in the active environment before running training."
	) from exc

from tribes_env import TribesEnv


ENGINEERED_FEATURE_DIM = 1024
RAW_FEATURE_DIM = 256
DEFAULT_ENCODER_MODE = "combined"


LOGGER = logging.getLogger(__name__)


@dataclass
class DQNConfig:
	level_file: str = "tribes/levels/SampleLevel.csv"
	game_mode: str = "SCORE"
	seed: int = 123
	compile_first: bool = True
	encoder_mode: str = DEFAULT_ENCODER_MODE
	engineered_dim: int = ENGINEERED_FEATURE_DIM
	raw_dim: int = RAW_FEATURE_DIM
	hidden_dim: int = 256
	learning_rate: float = 3e-4
	gamma: float = 0.99
	batch_size: int = 64
	replay_size: int = 50_000
	learning_starts: int = 1_000
	train_frequency: int = 1
	target_update_frequency: int = 500
	gradient_clip_norm: float = 5.0
	epsilon_start: float = 1.0
	epsilon_final: float = 0.10
	epsilon_decay_steps: int = 20_000
	reward_scale: float = 50.0
	train_episodes: int = 100
	eval_episodes: int = 5
	eval_every_episodes: int = 10
	eval_seed_offset: int = 10_000
	max_steps_per_episode: int = 2_000
	total_train_steps: int = 0
	checkpoint_path: str = "checkpoints/dqn.pt"
	resume: bool = False
	device: str = "cpu"


@dataclass
class Transition:
	state: np.ndarray
	action: int
	reward: float
	next_state: np.ndarray
	done: bool
	next_action_mask: np.ndarray


class ReplayBuffer:
	def __init__(self, capacity: int) -> None:
		self.capacity = int(capacity)
		self._buffer: list[Transition | None] = [None] * self.capacity
		self._index = 0
		self._size = 0

	def __len__(self) -> int:
		return self._size

	def add(self, transition: Transition) -> None:
		self._buffer[self._index] = transition
		self._index = (self._index + 1) % self.capacity
		self._size = min(self._size + 1, self.capacity)

	def sample(self, batch_size: int) -> list[Transition]:
		if batch_size > self._size:
			raise ValueError("Cannot sample more transitions than currently stored")
		indices = random.sample(range(self._size), batch_size)
		return [self._buffer[index] for index in indices if self._buffer[index] is not None]


class JsonStateEncoder:
	def __init__(self, mode: str = DEFAULT_ENCODER_MODE, engineered_dim: int = ENGINEERED_FEATURE_DIM, raw_dim: int = RAW_FEATURE_DIM) -> None:
		self.mode = mode
		self.engineered_dim = int(engineered_dim)
		self.raw_dim = int(raw_dim)

	@property
	def output_dim(self) -> int:
		if self.mode == "engineered":
			return self.engineered_dim
		if self.mode == "raw":
			return self.raw_dim
		if self.mode == "combined":
			return self.engineered_dim + self.raw_dim
		raise ValueError(f"Unknown encoder mode: {self.mode}")

	def encode(self, state_json: str) -> np.ndarray:
		if self.mode == "engineered":
			return self._encode_engineered(state_json)
		if self.mode == "raw":
			return self._encode_raw(state_json)
		if self.mode == "combined":
			return np.concatenate([self._encode_engineered(state_json), self._encode_raw(state_json)], axis=0)
		raise ValueError(f"Unknown encoder mode: {self.mode}")

	def _encode_engineered(self, state_json: str) -> np.ndarray:
		vector = np.zeros(self.engineered_dim, dtype=np.float32)
		try:
			state = json.loads(state_json)
		except json.JSONDecodeError:
			return vector

		features: list[float] = []

		def add(value: float) -> None:
			if len(features) < self.engineered_dim:
				features.append(float(value))

		def scaled(value: float, scale: float = 10.0) -> float:
			return math.tanh(float(value) / float(scale))

		def safe_number(value: Any, default: float = 0.0) -> float:
			if value is None:
				return default
			if isinstance(value, bool):
				return 1.0 if value else 0.0
			if isinstance(value, (int, float, np.integer, np.floating)):
				return float(value)
			return default

		def encode_string(text: Any) -> None:
			string_value = str(text)
			add(len(string_value) / 32.0)
			add(((zlib.crc32(string_value.encode("utf-8")) % 4096) / 2048.0) - 1.0)

		def encode_stats(values: Iterable[float], *, scale: float = 10.0) -> None:
			data = [float(value) for value in values]
			if not data:
				add(0.0)
				add(0.0)
				add(0.0)
				add(0.0)
				return
			array = np.asarray(data, dtype=np.float32)
			add(float(array.mean()) / scale)
			add(float(array.std()) / scale)
			add(float(array.min()) / scale)
			add(float(array.max()) / scale)

		def encode_grid(grid: Any, *, histogram_max: int = 16) -> None:
			if not isinstance(grid, list) or not grid:
				add(0.0)
				add(0.0)
				add(0.0)
				add(0.0)
				add(0.0)
				add(0.0)
				return
			flattened: list[float] = []
			for row in grid:
				if isinstance(row, list):
					flattened.extend(float(value) for value in row if isinstance(value, (int, float, np.integer, np.floating, bool)))
				elif isinstance(row, (int, float, np.integer, np.floating, bool)):
					flattened.append(float(row))
			if not flattened:
				add(0.0)
				add(0.0)
				add(0.0)
				add(0.0)
				add(0.0)
				add(0.0)
				return
			array = np.asarray(flattened, dtype=np.float32)
			add(float((array != 0).mean()))
			add(float(array.mean()) / 10.0)
			add(float(array.std()) / 10.0)
			add(float(array.min()) / 10.0)
			add(float(array.max()) / 10.0)
			add(float(len(np.unique(array))) / 32.0)
			for value in range(histogram_max):
				add(float(np.sum(array == value)) / max(1.0, float(array.size)))

		def encode_board(board: dict[str, Any]) -> None:
			encode_grid(board.get("terrain"))
			encode_grid(board.get("resource"))
			encode_grid(board.get("building"))
			encode_grid(board.get("unitID"))
			encode_grid(board.get("cityID"))
			encode_grid(board.get("network"), histogram_max=2)
			add(safe_number(board.get("actorIDcounter")) / 32.0)

		def encode_active_tribe(tribe: dict[str, Any]) -> None:
			add(safe_number(tribe.get("score")) / 100.0)
			add(safe_number(tribe.get("star")) / 20.0)
			add(safe_number(tribe.get("nKills")) / 20.0)
			add(safe_number(tribe.get("nPacifistCount")) / 20.0)
			add(safe_number(len(tribe.get("citiesID", []))) / 20.0)
			add(safe_number(len(tribe.get("extraUnits", []))) / 20.0)
			add(safe_number(len(tribe.get("monuments", []))) / 20.0)
			add(safe_number(len(tribe.get("tribesMet", []))) / 20.0)
			add(1.0 if tribe.get("winner") == 0 else 0.0)
			add(1.0 if tribe.get("winner") == 1 else 0.0)
			add(1.0 if tribe.get("winner") == -1 else 0.0)
			technology = tribe.get("technology", {})
			researched = technology.get("researched", []) if isinstance(technology, dict) else []
			add(float(sum(bool(item) for item in researched)) / max(1.0, float(len(researched))))
			add(1.0 if isinstance(technology, dict) and technology.get("everythingResearched") else 0.0)
			encode_string(tribe.get("type"))

		def encode_units(units: dict[str, Any]) -> None:
			if not isinstance(units, dict) or not units:
				for _ in range(14):
					add(0.0)
				return
			current_hp = []
			veteran = []
			kills = []
			xs = []
			ys = []
			city_ids = []
			tribe_ids = []
			for unit in units.values():
				if not isinstance(unit, dict):
					continue
				current_hp.append(safe_number(unit.get("currentHP")))
				veteran.append(1.0 if unit.get("isVeteran") else 0.0)
				kills.append(safe_number(unit.get("kill")))
				xs.append(safe_number(unit.get("x")))
				ys.append(safe_number(unit.get("y")))
				city_ids.append(safe_number(unit.get("cityID")))
				tribe_ids.append(safe_number(unit.get("tribeId")))
			add(float(len(units)) / 20.0)
			encode_stats(current_hp, scale=10.0)
			encode_stats(veteran, scale=1.0)
			encode_stats(kills, scale=10.0)
			encode_stats(xs, scale=16.0)
			encode_stats(ys, scale=16.0)
			encode_stats(city_ids, scale=10.0)
			encode_stats(tribe_ids, scale=10.0)

		def encode_cities(cities: dict[str, Any]) -> None:
			if not isinstance(cities, dict) or not cities:
				for _ in range(16):
					add(0.0)
				return
			level = []
			population = []
			production = []
			walls = []
			capital = []
			xs = []
			ys = []
			units = []
			for city in cities.values():
				if not isinstance(city, dict):
					continue
				level.append(safe_number(city.get("level")))
				population.append(safe_number(city.get("population")))
				production.append(safe_number(city.get("production")))
				walls.append(1.0 if city.get("hasWalls") else 0.0)
				capital.append(1.0 if city.get("isCapital") else 0.0)
				xs.append(safe_number(city.get("x")))
				ys.append(safe_number(city.get("y")))
				units.append(safe_number(len(city.get("units", []))))
			add(float(len(cities)) / 20.0)
			encode_stats(level, scale=10.0)
			encode_stats(population, scale=10.0)
			encode_stats(production, scale=10.0)
			encode_stats(walls, scale=1.0)
			encode_stats(capital, scale=1.0)
			encode_stats(xs, scale=16.0)
			encode_stats(ys, scale=16.0)
			encode_stats(units, scale=10.0)

		def encode_all_tribes(all_tribes: dict[str, Any], active_tribe_id: str) -> None:
			if not isinstance(all_tribes, dict) or not all_tribes:
				for _ in range(24):
					add(0.0)
				return
			scores: list[float] = []
			stars: list[float] = []
			city_counts: list[float] = []
			unit_counts: list[float] = []
			tech_completion: list[float] = []
			active_score = 0.0
			active_stars = 0.0
			for tribe_id, tribe in all_tribes.items():
				if not isinstance(tribe, dict):
					continue
				score = safe_number(tribe.get("score"))
				star = safe_number(tribe.get("star"))
				city_count = float(len(tribe.get("citiesID", [])))
				unit_count = float(len(tribe.get("extraUnits", [])))
				technology = tribe.get("technology", {})
				researched = technology.get("researched", []) if isinstance(technology, dict) else []
				completion = float(sum(bool(item) for item in researched)) / max(1.0, float(len(researched)))
				scores.append(score)
				stars.append(star)
				city_counts.append(city_count)
				unit_counts.append(unit_count)
				tech_completion.append(completion)
				if str(tribe_id) == active_tribe_id:
					active_score = score
					active_stars = star
			add(float(len(scores)) / 8.0)
			encode_stats(scores, scale=100.0)
			encode_stats(stars, scale=20.0)
			encode_stats(city_counts, scale=20.0)
			encode_stats(unit_counts, scale=20.0)
			encode_stats(tech_completion, scale=1.0)
			best_score = max(scores)
			mean_score = float(np.mean(scores))
			add((active_score - mean_score) / 100.0)
			add((active_score - best_score) / 100.0)
			add(active_stars / 20.0)
			add(1.0 if active_score >= best_score else 0.0)

		board = state.get("board", {})
		tribes = state.get("tribes", {})
		active_tribe_id = str(state.get("activeTribeID"))
		active_tribe = tribes.get(active_tribe_id, {}) if isinstance(tribes, dict) else {}

		add(safe_number(state.get("tick")) / 100.0)
		add(1.0 if state.get("gameIsOver") else 0.0)
		encode_string(state.get("gameMode"))
		encode_board(board if isinstance(board, dict) else {})
		encode_all_tribes(tribes if isinstance(tribes, dict) else {}, active_tribe_id)
		encode_active_tribe(active_tribe if isinstance(active_tribe, dict) else {})
		encode_units(state.get("unit", {}))
		encode_cities(state.get("city", {}))

		numeric = np.asarray(features, dtype=np.float32)
		vector[: len(numeric)] = numeric
		if len(numeric) < self.engineered_dim:
			vector[len(numeric) :] = 0.0
		return vector

	def _encode_raw(self, state_json: str) -> np.ndarray:
		vector = np.zeros(self.raw_dim, dtype=np.float32)
		if not state_json:
			return vector
		text = state_json.lower()
		window = 3
		if len(text) < window:
			index = zlib.crc32(text.encode("utf-8")) % self.raw_dim
			vector[index] += 1.0
			return vector
		for start in range(len(text) - window + 1):
			ngram = text[start : start + window]
			index = zlib.crc32(ngram.encode("utf-8")) % self.raw_dim
			vector[index] += 1.0
		vector /= max(1.0, float(np.linalg.norm(vector)))
		return vector.astype(np.float32, copy=False)


class QNetwork(nn.Module):
	def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class DQNAgent:
	def __init__(self, state_dim: int, action_dim: int, config: DQNConfig) -> None:
		self.state_dim = int(state_dim)
		self.action_dim = int(action_dim)
		self.config = config
		self.device = torch.device(config.device)
		self.online = QNetwork(self.state_dim, self.action_dim, config.hidden_dim).to(self.device)
		self.target = QNetwork(self.state_dim, self.action_dim, config.hidden_dim).to(self.device)
		self.target.load_state_dict(self.online.state_dict())
		self.target.eval()
		self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
		self.train_steps = 0
		self.episodes = 0

	def select_action(self, state: np.ndarray, action_mask: np.ndarray, epsilon: float, greedy: bool = False) -> int:
		valid_actions = np.flatnonzero(action_mask.astype(bool))
		if valid_actions.size == 0:
			raise RuntimeError("No valid actions available for the current state")

		if (not greedy) and random.random() < epsilon:
			return int(np.random.choice(valid_actions))

		state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
		with torch.no_grad():
			q_values = self.online(state_tensor).squeeze(0)
			masked_q = self._masked_q_values(q_values, action_mask)
			return int(torch.argmax(masked_q).item())

	def train_batch(self, batch: list[Transition]) -> float:
		states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device)
		actions = torch.as_tensor([item.action for item in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
		rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
		next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32, device=self.device)
		dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device)
		next_action_masks = torch.as_tensor(np.stack([item.next_action_mask for item in batch]), dtype=torch.bool, device=self.device)

		q_values = self.online(states).gather(1, actions).squeeze(1)
		with torch.no_grad():
			next_online = self.online(next_states)
			next_online = self._mask_tensor_q_values(next_online, next_action_masks)
			next_actions = torch.argmax(next_online, dim=1, keepdim=True)
			next_target = self.target(next_states).gather(1, next_actions).squeeze(1)
			targets = rewards + self.config.gamma * (1.0 - dones) * next_target

		loss = torch.nn.functional.smooth_l1_loss(q_values, targets)

		self.optimizer.zero_grad(set_to_none=True)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip_norm)
		self.optimizer.step()
		self.train_steps += 1
		return float(loss.item())

	def update_target(self) -> None:
		self.target.load_state_dict(self.online.state_dict())

	def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
		checkpoint = {
			"config": asdict(self.config),
			"state_dim": self.state_dim,
			"action_dim": self.action_dim,
			"online_state_dict": self.online.state_dict(),
			"target_state_dict": self.target.state_dict(),
			"optimizer_state_dict": self.optimizer.state_dict(),
			"train_steps": self.train_steps,
			"episodes": self.episodes,
			"extra": extra or {},
		}
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(checkpoint, path)

	@classmethod
	def load(cls, path: str | Path, config: DQNConfig | None = None) -> tuple["DQNAgent", dict[str, Any]]:
		checkpoint = torch.load(Path(path), map_location="cpu")
		loaded_config = DQNConfig(**checkpoint["config"])
		if config is not None:
			structural_fields = ("encoder_mode", "engineered_dim", "raw_dim", "hidden_dim")
			for field_name in structural_fields:
				if getattr(config, field_name) != getattr(loaded_config, field_name):
					raise ValueError(
						f"Checkpoint config mismatch for {field_name}: "
						f"expected {getattr(loaded_config, field_name)!r}, got {getattr(config, field_name)!r}"
					)
			loaded_config = config
		agent = cls(checkpoint["state_dim"], checkpoint["action_dim"], loaded_config)
		agent.online.load_state_dict(checkpoint["online_state_dict"])
		agent.target.load_state_dict(checkpoint["target_state_dict"])
		agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		agent.train_steps = int(checkpoint.get("train_steps", 0))
		agent.episodes = int(checkpoint.get("episodes", 0))
		return agent, checkpoint.get("extra", {})

	def _masked_q_values(self, q_values: torch.Tensor, action_mask: np.ndarray) -> torch.Tensor:
		mask = torch.as_tensor(action_mask.astype(bool), dtype=torch.bool, device=q_values.device)
		return q_values.masked_fill(~mask, torch.finfo(q_values.dtype).min)

	def _mask_tensor_q_values(self, q_values: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
		return q_values.masked_fill(~action_masks, torch.finfo(q_values.dtype).min)


def epsilon_by_step(step: int, config: DQNConfig) -> float:
	if config.epsilon_decay_steps <= 0:
		return config.epsilon_final
	progress = min(1.0, step / float(config.epsilon_decay_steps))
	return config.epsilon_start + progress * (config.epsilon_final - config.epsilon_start)


def make_env(config: DQNConfig) -> TribesEnv:
	return TribesEnv(
		level_file=config.level_file,
		game_mode=config.game_mode,
		seed=config.seed,
		compile_first=config.compile_first,
	)


def evaluate_agent(agent: DQNAgent, config: DQNConfig, encoder: JsonStateEncoder, episodes: int, seed_offset: int = 0) -> dict[str, Any]:
	env = make_env(config)
	results: dict[str, Any] = {"episode_rewards": []}
	for episode in range(episodes):
		obs, info = env.reset(seed=config.seed + seed_offset + episode)
		state = encoder.encode(obs["state_json"])
		episode_reward = 0.0
		episode_steps = 0
		for _ in range(config.max_steps_per_episode):
			action_mask = np.asarray(info["action_mask"], dtype=np.bool_)
			action = agent.select_action(state, action_mask, epsilon=0.0, greedy=True)
			next_obs, reward, terminated, truncated, next_info = env.step(action)
			state = encoder.encode(next_obs["state_json"])
			info = next_info
			episode_reward += float(reward)
			episode_steps += 1
			if terminated or truncated:
				break
		results["episode_rewards"].append(episode_reward)
		LOGGER.info("eval episode=%d reward=%.3f steps=%d", episode + 1, episode_reward, episode_steps)
	env.close()
	return results


def run_training(config: DQNConfig) -> dict[str, Any]:
	random.seed(config.seed)
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)

	env = make_env(config)
	encoder = JsonStateEncoder(config.encoder_mode, config.engineered_dim, config.raw_dim)
	replay = ReplayBuffer(config.replay_size)
	stats: dict[str, Any] = {"episode_rewards": [], "episode_losses": []}
	global_step = 0
	total_updates = 0

	obs, info = env.reset(seed=config.seed)
	state = encoder.encode(obs["state_json"])
	state_dim = state.shape[0]
	agent: DQNAgent | None = None
	checkpoint_path = Path(config.checkpoint_path)
	best_eval_reward = float("-inf")
	if config.resume and checkpoint_path.exists():
		LOGGER.info("resuming from checkpoint=%s", checkpoint_path)
		agent, extra = DQNAgent.load(checkpoint_path, config=config)
		stats.update(extra.get("stats", {}))
		global_step = int(extra.get("global_step", 0))
		total_updates = int(extra.get("total_updates", 0))
		best_eval_reward = float(extra.get("best_eval_reward", best_eval_reward))
	else:
		agent = DQNAgent(state_dim, env.action_space.n, config)

	assert agent is not None
	agent.online.train()

	for episode in range(config.train_episodes):
		obs, info = env.reset(seed=config.seed + episode)
		state = encoder.encode(obs["state_json"])
		episode_reward = 0.0
		episode_loss_values: list[float] = []
		episode_steps = 0

		for _ in range(config.max_steps_per_episode):
			action_mask = np.asarray(info["action_mask"], dtype=np.bool_)
			epsilon = epsilon_by_step(global_step, config)
			action = agent.select_action(state, action_mask, epsilon)
			next_obs, reward, terminated, truncated, next_info = env.step(action)
			next_state = encoder.encode(next_obs["state_json"])
			next_mask = np.asarray(next_info["action_mask"], dtype=np.bool_)

			replay.add(
				Transition(
					state=state,
					action=action,
					reward=float(np.clip(reward / max(1.0, config.reward_scale), -1.0, 1.0)),
					next_state=next_state,
					done=bool(terminated or truncated),
					next_action_mask=next_mask,
				)
			)

			state = next_state
			info = next_info
			episode_reward += float(reward)
			episode_steps += 1
			global_step += 1

			if len(replay) >= config.batch_size and global_step >= config.learning_starts and global_step % config.train_frequency == 0:
				batch = replay.sample(config.batch_size)
				loss = agent.train_batch(batch)
				episode_loss_values.append(loss)
				total_updates += 1
				if total_updates % config.target_update_frequency == 0:
					agent.update_target()

			if terminated or truncated:
				break

		agent.episodes += 1
		stats["episode_rewards"].append(episode_reward)
		stats["episode_losses"].append(float(np.mean(episode_loss_values)) if episode_loss_values else None)
		avg_loss = float(np.mean(episode_loss_values)) if episode_loss_values else None
		epsilon = epsilon_by_step(global_step, config)
		eval_reward = None
		if config.eval_every_episodes > 0 and (episode + 1) % config.eval_every_episodes == 0:
			eval_results = evaluate_agent(agent, config, encoder, config.eval_episodes, seed_offset=config.eval_seed_offset + episode * 100)
			eval_reward = float(np.mean(eval_results["episode_rewards"])) if eval_results["episode_rewards"] else None
			if eval_reward is not None and eval_reward > best_eval_reward:
				best_eval_reward = eval_reward
				agent.save(
					checkpoint_path.with_suffix(".best.pt"),
					extra={
						"global_step": global_step,
						"total_updates": total_updates,
						"stats": stats,
						"best_eval_reward": best_eval_reward,
					},
				)
		LOGGER.info(
			"train episode=%d reward=%.3f steps=%d loss=%s epsilon=%.4f updates=%d eval_reward=%s best_eval=%s",
			episode + 1,
			episode_reward,
			episode_steps,
			"none" if avg_loss is None else f"{avg_loss:.6f}",
			epsilon,
			total_updates,
			"none" if eval_reward is None else f"{eval_reward:.3f}",
			"none" if best_eval_reward == float("-inf") else f"{best_eval_reward:.3f}",
		)
		agent.save(
			checkpoint_path,
			extra={
				"global_step": global_step,
				"total_updates": total_updates,
				"stats": stats,
				"best_eval_reward": best_eval_reward,
			},
		)

	env.close()
	return stats


def run_evaluation(config: DQNConfig) -> dict[str, Any]:
	random.seed(config.seed)
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)

	encoder = JsonStateEncoder(config.encoder_mode, config.engineered_dim, config.raw_dim)
	checkpoint_path = Path(config.checkpoint_path)
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
	agent, _ = DQNAgent.load(checkpoint_path, config=config)
	agent.online.eval()
	results = evaluate_agent(agent, config, encoder, config.eval_episodes)
	return results


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Deep Q-Network for TribesEnv")
	subparsers = parser.add_subparsers(dest="command", required=True)

	def add_shared_arguments(subparser: argparse.ArgumentParser) -> None:
		subparser.add_argument("--level-file", default="tribes/levels/SampleLevel.csv")
		subparser.add_argument("--game-mode", default="SCORE")
		subparser.add_argument("--seed", type=int, default=123)
		subparser.add_argument("--checkpoint-path", default="checkpoints/dqn.pt")
		subparser.add_argument("--encoder-mode", choices=("engineered", "raw", "combined"), default=DEFAULT_ENCODER_MODE)
		subparser.add_argument("--engineered-dim", type=int, default=ENGINEERED_FEATURE_DIM)
		subparser.add_argument("--raw-dim", type=int, default=RAW_FEATURE_DIM)
		subparser.add_argument("--hidden-dim", type=int, default=256)
		subparser.add_argument("--compile-first", action=argparse.BooleanOptionalAction, default=True)
		subparser.add_argument("--max-steps-per-episode", type=int, default=2_000)

	train = subparsers.add_parser("train", help="Train a DQN agent")
	add_shared_arguments(train)
	train.add_argument("--train-episodes", type=int, default=100)
	train.add_argument("--batch-size", type=int, default=64)
	train.add_argument("--replay-size", type=int, default=50_000)
	train.add_argument("--learning-starts", type=int, default=1_000)
	train.add_argument("--train-frequency", type=int, default=1)
	train.add_argument("--target-update-frequency", type=int, default=1_000)
	train.add_argument("--eval-every-episodes", type=int, default=10)
	train.add_argument("--eval-episodes-during-train", type=int, default=1)
	train.add_argument("--eval-seed-offset", type=int, default=10_000)
	train.add_argument("--learning-rate", type=float, default=1e-4)
	train.add_argument("--gamma", type=float, default=0.99)
	train.add_argument("--epsilon-start", type=float, default=1.0)
	train.add_argument("--epsilon-final", type=float, default=0.10)
	train.add_argument("--epsilon-decay-steps", type=int, default=20_000)
	train.add_argument("--reward-scale", type=float, default=50.0)
	train.add_argument("--gradient-clip-norm", type=float, default=5.0)
	train.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
	train.add_argument("--device", default="cpu")

	eval_parser = subparsers.add_parser("eval", help="Evaluate a trained DQN agent")
	add_shared_arguments(eval_parser)
	eval_parser.add_argument("--eval-episodes", type=int, default=5)
	eval_parser.add_argument("--device", default="cpu")

	return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> DQNConfig:
	config = DQNConfig(
		level_file=args.level_file,
		game_mode=args.game_mode,
		seed=args.seed,
		compile_first=args.compile_first,
		encoder_mode=args.encoder_mode,
		engineered_dim=args.engineered_dim,
		raw_dim=args.raw_dim,
		hidden_dim=args.hidden_dim,
		checkpoint_path=args.checkpoint_path,
		max_steps_per_episode=args.max_steps_per_episode,
		device=getattr(args, "device", "cpu"),
	)
	if args.command == "train":
		config.batch_size = args.batch_size
		config.replay_size = args.replay_size
		config.learning_starts = args.learning_starts
		config.train_frequency = args.train_frequency
		config.target_update_frequency = args.target_update_frequency
		config.eval_every_episodes = args.eval_every_episodes
		config.eval_episodes = args.eval_episodes_during_train
		config.eval_seed_offset = args.eval_seed_offset
		config.learning_rate = args.learning_rate
		config.gamma = args.gamma
		config.epsilon_start = args.epsilon_start
		config.epsilon_final = args.epsilon_final
		config.epsilon_decay_steps = args.epsilon_decay_steps
		config.reward_scale = args.reward_scale
		config.gradient_clip_norm = args.gradient_clip_norm
		config.train_episodes = args.train_episodes
		config.resume = args.resume
	elif args.command == "eval":
		config.eval_episodes = args.eval_episodes
	return config


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(message)s")
	args = parse_args()
	config = config_from_args(args)
	if args.command == "train":
		stats = run_training(config)
		print(json.dumps(stats, indent=2, sort_keys=True))
	elif args.command == "eval":
		results = run_evaluation(config)
		print(json.dumps(results, indent=2, sort_keys=True))
	else:
		raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
	main()
