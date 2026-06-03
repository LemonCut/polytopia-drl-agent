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
import time

try:
	import torch
	from torch import nn
except Exception as exc:  # pragma: no cover - imported at runtime
	raise RuntimeError(
		"dqn.py requires PyTorch. Install torch in the active environment before running training."
	) from exc

from tribes_env import TribesEnv
from vec_env import VecEnv
from encoder import JsonStateEncoder


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
	batch_size: int = 512
	replay_size: int = 100_000
	learning_starts: int = 2_000
	train_frequency: int = 4
	target_update_frequency: int = 1_000
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
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	num_workers: int = 1


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
		# Automatic mixed precision (AMP) support
		self.use_amp = torch.cuda.is_available()
		# Use torch.amp API (preferred) when available
		self.scaler: torch.amp.GradScaler | None = (
			torch.amp.GradScaler("cuda") if self.use_amp else None
		)
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
		# Prepare tensors using pinned CPU memory and async transfer to GPU where possible
		states_np = np.stack([item.state for item in batch]).astype(np.float32, copy=False)
		actions_np = np.asarray([item.action for item in batch], dtype=np.int64)
		rewards_np = np.asarray([item.reward for item in batch], dtype=np.float32)
		next_states_np = np.stack([item.next_state for item in batch]).astype(np.float32, copy=False)
		dones_np = np.asarray([item.done for item in batch], dtype=np.float32)
		next_action_masks_np = np.stack([item.next_action_mask for item in batch]).astype(np.bool_)

		use_pin = torch.cuda.is_available()
		if use_pin:
			states = torch.from_numpy(states_np).pin_memory()
			actions = torch.from_numpy(actions_np).pin_memory().unsqueeze(1)
			rewards = torch.from_numpy(rewards_np).pin_memory()
			next_states = torch.from_numpy(next_states_np).pin_memory()
			dones = torch.from_numpy(dones_np).pin_memory()
			next_action_masks = torch.from_numpy(next_action_masks_np).pin_memory()
		else:
			states = torch.from_numpy(states_np)
			actions = torch.from_numpy(actions_np).unsqueeze(1)
			rewards = torch.from_numpy(rewards_np)
			next_states = torch.from_numpy(next_states_np)
			dones = torch.from_numpy(dones_np)
			next_action_masks = torch.from_numpy(next_action_masks_np)

		# Move to device (non_blocking when pinned and CUDA available)
		non_blocking = True if use_pin and self.device.type == "cuda" else False
		states = states.to(self.device, non_blocking=non_blocking)
		actions = actions.to(self.device, non_blocking=non_blocking)
		rewards = rewards.to(self.device, non_blocking=non_blocking)
		next_states = next_states.to(self.device, non_blocking=non_blocking)
		dones = dones.to(self.device, non_blocking=non_blocking)
		next_action_masks = next_action_masks.to(self.device, non_blocking=non_blocking)
		# Forward and loss with optional AMP
		if self.use_amp and self.scaler is not None:
			with torch.amp.autocast(device_type="cuda"):
				q_values = self.online(states).gather(1, actions).squeeze(1)
				with torch.no_grad():
					next_online = self.online(next_states)
					next_online = self._mask_tensor_q_values(next_online, next_action_masks)
					next_actions = torch.argmax(next_online, dim=1, keepdim=True)
					next_target = self.target(next_states).gather(1, next_actions).squeeze(1)
					targets = rewards + self.config.gamma * (1.0 - dones) * next_target
				loss = torch.nn.functional.smooth_l1_loss(q_values, targets)

			self.optimizer.zero_grad(set_to_none=True)
			self.scaler.scale(loss).backward()
			torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip_norm)
			self.scaler.step(self.optimizer)
			self.scaler.update()
			self.train_steps += 1
			return float(loss.item())
		else:
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
			"scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
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
		# Restore AMP scaler state if present
		scaler_state = checkpoint.get("scaler_state_dict")
		if scaler_state is not None and agent.scaler is not None:
			agent.scaler.load_state_dict(scaler_state)
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
	# Single env used for action space info; main parallel runner will use VecEnv when num_workers>1
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
		episode_elapsed = None
		# Parallel multi-worker loop
		if config.num_workers > 1:
			vec = VecEnv(
				config.num_workers,
				env_kwargs={
				"level_file": config.level_file,
				"game_mode": config.game_mode,
				"compile_first": config.compile_first,
				"max_episode_steps": config.max_steps_per_episode,
				},
				encoder_kwargs={
					"mode": config.encoder_mode,
					"engineered_dim": config.engineered_dim,
					"raw_dim": config.raw_dim,
				},
			)
			# initialize workers
			seeds = [config.seed + i for i in range(config.num_workers)]
			results = vec.reset(seeds=seeds)
			states = [obs for obs, info in results]
			infos = [info for obs, info in results]
			# per-worker episode start times
			episode_start_times = [time.perf_counter() for _ in range(config.num_workers)]
			episode_rewards = [0.0] * config.num_workers
			episode_steps = [0] * config.num_workers
			episode_loss_values: list[float] = []
			episodes_completed = 0
			# run until we've completed the requested number of episodes across workers
			while episodes_completed < config.train_episodes:
				# select actions for all workers
				actions = []
				for w in range(config.num_workers):
					action_mask = np.asarray(infos[w]["action_mask"], dtype=np.bool_)
					epsilon = epsilon_by_step(global_step, config)
					actions.append(agent.select_action(states[w], action_mask, epsilon))

				# step all workers in parallel
				results = vec.step(actions)
				for w, (next_obs, reward, terminated, truncated, next_info) in enumerate(results):
					next_state = next_obs
					next_mask = np.asarray(next_info["action_mask"], dtype=np.bool_)

					replay.add(
						Transition(
							state=states[w],
							action=actions[w],
							reward=float(np.clip(reward / max(1.0, config.reward_scale), -1.0, 1.0)),
							next_state=next_state,
							done=bool(terminated or truncated),
							next_action_mask=next_mask,
						)
					)

					states[w] = next_state
					infos[w] = next_info
					episode_rewards[w] += float(reward)
					episode_steps[w] += 1
					global_step += 1

					# training step
					if len(replay) >= config.batch_size and global_step >= config.learning_starts and global_step % config.train_frequency == 0:
						batch = replay.sample(config.batch_size)
						loss = agent.train_batch(batch)
						episode_loss_values.append(loss)
						total_updates += 1
						if total_updates % config.target_update_frequency == 0:
							agent.update_target()

					if terminated or truncated:
						# finalize episode for this worker
						episodes_completed += 1
						avg_loss = float(np.mean(episode_loss_values)) if episode_loss_values else None
						stats["episode_rewards"].append(episode_rewards[w])
						stats["episode_losses"].append(avg_loss)
						# compute elapsed for this worker's episode
						elapsed = time.perf_counter() - episode_start_times[w]
						LOGGER.info(
							"train episode=%d worker=%d reward=%.3f steps=%d loss=%s elapsed=%.2fs",
							episodes_completed,
							w,
							episode_rewards[w],
							episode_steps[w],
							"none" if avg_loss is None else f"{avg_loss:.6f}",
							elapsed,
						)
						# reset this worker and restart its timer
						seed = config.seed + episodes_completed * 100 + w
						obs, info = vec.reset_worker(w, seed=seed)
						states[w] = obs
						infos[w] = info
						episode_rewards[w] = 0.0
						episode_steps[w] = 0
						episode_start_times[w] = time.perf_counter()

			# save checkpoint periodically (after the while loop iteration)
			agent.save(
				checkpoint_path,
				extra={
					"global_step": global_step,
					"total_updates": total_updates,
					"stats": stats,
					"best_eval_reward": best_eval_reward,
				},
			)
			vec.close()
		else:
			# Single-worker (original) loop
			obs, info = env.reset(seed=config.seed + episode)
			state = encoder.encode(obs["state_json"])
			episode_reward = 0.0
			episode_loss_values: list[float] = []
			episode_steps = 0
			# mark episode start time
			episode_start = time.perf_counter()

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
			# compute elapsed for this episode
			episode_elapsed = time.perf_counter() - episode_start

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
		elapsed_str = "none" if episode_elapsed is None else f"{episode_elapsed:.2f}s"
		LOGGER.info(
			"train episode=%d reward=%.3f steps=%d loss=%s epsilon=%.4f updates=%d eval_reward=%s best_eval=%s elapsed=%s",
			episode + 1,
			episode_reward,
			episode_steps,
			"none" if avg_loss is None else f"{avg_loss:.6f}",
			epsilon,
			total_updates,
			"none" if eval_reward is None else f"{eval_reward:.3f}",
			"none" if best_eval_reward == float("-inf") else f"{best_eval_reward:.3f}",
			elapsed_str,
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
		default_device = "cuda" if torch.cuda.is_available() else "cpu"
		subparser.add_argument("--device", default=default_device)
		subparser.add_argument("--num-workers", type=int, default=1)

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

	eval_parser = subparsers.add_parser("eval", help="Evaluate a trained DQN agent")
	add_shared_arguments(eval_parser)
	eval_parser.add_argument("--eval-episodes", type=int, default=5)

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
		num_workers=getattr(args, "num_workers", 1),
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
		print(f"Using device: {config.device}")
		stats = run_training(config)
		print(json.dumps(stats, indent=2, sort_keys=True))
	elif args.command == "eval":
		print(f"Using device: {config.device}")
		results = run_evaluation(config)
		print(json.dumps(results, indent=2, sort_keys=True))
	else:
		raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
	main()
