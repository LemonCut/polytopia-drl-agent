from __future__ import annotations

import argparse
import json
import logging
import random
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from encoder import DEFAULT_ENCODER_MODE, ENGINEERED_FEATURE_DIM, RAW_FEATURE_DIM, JsonStateEncoder
from tribes_env import TribesEnv


LOGGER = logging.getLogger(__name__)


@dataclass
class RankDistillConfig:
	level_file: str = "tribes/levels/SampleLevel.csv"
	game_mode: str = "SCORE"
	seed: int = 123
	compile_first: bool = True
	encoder_mode: str = "engineered"
	engineered_dim: int = ENGINEERED_FEATURE_DIM
	raw_dim: int = RAW_FEATURE_DIM
	action_feature_dim: int = 192
	hidden_dim: int = 256
	learning_rate: float = 3e-4
	batch_size: int = 64
	train_epochs_per_collection: int = 4
	expert_replay_size: int = 4096
	total_iterations: int = 40
	collection_steps: int = 256
	max_steps_per_episode: int = 2_000
	expert_type: str = "pMCTS"
	expert_fm_calls: int = 250
	expert_rollout_length: int = 12
	expert_rollouts: bool = False
	expert_time_millis: int = 10_000
	eval_every_iterations: int = 5
	eval_episodes: int = 5
	eval_max_steps_per_episode: int = 512
	eval_seed_offset: int = 10_000
	checkpoint_path: str = "checkpoints/expert_rank_distill.pt"
	resume: bool = False
	device: str = "cpu"


@dataclass
class RankSample:
	state: np.ndarray
	action_features: np.ndarray
	expert_local_index: int


class RankBatch:
	def __init__(self) -> None:
		self.samples: list[RankSample] = []

	def __len__(self) -> int:
		return len(self.samples)

	def extend(self, samples: list[RankSample], capacity: int) -> None:
		self.samples.extend(samples)
		if capacity > 0 and len(self.samples) > capacity:
			del self.samples[: len(self.samples) - capacity]


class ActionTextEncoder:
	ACTION_TYPES = (
		"RESEARCH_TECH",
		"END_TURN",
		"MOVE",
		"ATTACK",
		"CAPTURE",
		"SPAWN",
		"BUILD",
		"LEVEL_UP",
		"RESOURCE_GATHERING",
		"RECOVER",
		"MAKE_VETERAN",
		"BUILD_ROAD",
		"CLEAR_FOREST",
		"BURN_FOREST",
		"GROW_FOREST",
		"DESTROY",
		"DISBAND",
		"UPGRADE",
		"CONVERT",
		"HEAL_OTHERS",
		"EXAMINE",
	)

	def __init__(self, output_dim: int) -> None:
		self.output_dim = int(output_dim)
		self.type_to_index = {name: index for index, name in enumerate(self.ACTION_TYPES)}

	def encode(self, action: dict[str, Any]) -> np.ndarray:
		vector = np.zeros(self.output_dim, dtype=np.float32)
		action_type = str(action.get("type", ""))
		text = str(action.get("text", "")).lower()
		if action_type in self.type_to_index and self.type_to_index[action_type] < self.output_dim:
			vector[self.type_to_index[action_type]] = 1.0
		offset = min(len(self.ACTION_TYPES), self.output_dim)
		buckets = max(1, self.output_dim - offset)
		for token in text.replace(":", " ").replace(",", " ").split():
			index = offset + (zlib.crc32(token.encode("utf-8")) % buckets)
			vector[index] += 1.0
		for start in range(max(0, len(text) - 2)):
			ngram = text[start : start + 3]
			index = offset + (zlib.crc32(ngram.encode("utf-8")) % buckets)
			vector[index] += 0.25
		norm = float(np.linalg.norm(vector))
		if norm > 0:
			vector /= norm
		return vector


class RankPolicy(nn.Module):
	def __init__(self, state_dim: int, action_feature_dim: int, hidden_dim: int) -> None:
		super().__init__()
		self.state_net = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.GELU(),
		)
		self.action_net = nn.Sequential(
			nn.Linear(action_feature_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.GELU(),
		)
		self.score = nn.Sequential(
			nn.Linear(hidden_dim * 3, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, 1),
		)

	def forward(self, states: torch.Tensor, action_features: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
		state_features = self.state_net(states)
		action_emb = self.action_net(action_features)
		state_expanded = state_features.unsqueeze(1).expand(-1, action_emb.shape[1], -1)
		x = torch.cat([state_expanded, action_emb, state_expanded * action_emb], dim=-1)
		scores = self.score(x).squeeze(-1)
		return scores.masked_fill(~action_mask, torch.finfo(scores.dtype).min)


class RankAgent:
	def __init__(self, state_dim: int, config: RankDistillConfig) -> None:
		self.config = config
		self.device = torch.device(config.device)
		self.model = RankPolicy(state_dim, config.action_feature_dim, config.hidden_dim).to(self.device)
		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
		self.iteration = 0
		self.best_eval_reward = float("-inf")

	@torch.no_grad()
	def select_action(self, state: np.ndarray, action_features: np.ndarray) -> int:
		self.model.eval()
		states, actions, mask, _ = collate_samples(
			[RankSample(state=state, action_features=action_features, expert_local_index=0)],
			self.device,
		)
		scores = self.model(states, actions, mask)
		return int(torch.argmax(scores, dim=1).item())

	def train_batch(self, samples: list[RankSample]) -> dict[str, float]:
		self.model.train()
		states, actions, mask, targets = collate_samples(samples, self.device)
		scores = self.model(states, actions, mask)
		loss = torch.nn.functional.cross_entropy(scores, targets)
		with torch.no_grad():
			acc = (torch.argmax(scores, dim=1) == targets).float().mean()
		self.optimizer.zero_grad(set_to_none=True)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
		self.optimizer.step()
		return {"loss": float(loss.item()), "accuracy": float(acc.item())}

	def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(
			{
				"config": asdict(self.config),
				"model_state_dict": self.model.state_dict(),
				"optimizer_state_dict": self.optimizer.state_dict(),
				"iteration": self.iteration,
				"best_eval_reward": self.best_eval_reward,
				"extra": extra or {},
			},
			path,
		)

	@classmethod
	def load(cls, path: str | Path, state_dim: int, config: RankDistillConfig) -> tuple["RankAgent", dict[str, Any]]:
		checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
		agent = cls(state_dim, config)
		agent.model.load_state_dict(checkpoint["model_state_dict"])
		agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		agent.iteration = int(checkpoint.get("iteration", 0))
		agent.best_eval_reward = float(checkpoint.get("best_eval_reward", float("-inf")))
		return agent, checkpoint.get("extra", {})


def collate_samples(samples: list[RankSample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	max_actions = max(sample.action_features.shape[0] for sample in samples)
	state_array = np.stack([sample.state for sample in samples])
	action_dim = samples[0].action_features.shape[1]
	action_array = np.zeros((len(samples), max_actions, action_dim), dtype=np.float32)
	mask_array = np.zeros((len(samples), max_actions), dtype=np.bool_)
	targets = np.zeros(len(samples), dtype=np.int64)
	for i, sample in enumerate(samples):
		n_actions = sample.action_features.shape[0]
		action_array[i, :n_actions] = sample.action_features
		mask_array[i, :n_actions] = True
		targets[i] = sample.expert_local_index
	return (
		torch.as_tensor(state_array, dtype=torch.float32, device=device),
		torch.as_tensor(action_array, dtype=torch.float32, device=device),
		torch.as_tensor(mask_array, dtype=torch.bool, device=device),
		torch.as_tensor(targets, dtype=torch.long, device=device),
	)


def make_env(config: RankDistillConfig) -> TribesEnv:
	return TribesEnv(
		level_file=config.level_file,
		game_mode=config.game_mode,
		seed=config.seed,
		max_episode_steps=config.max_steps_per_episode,
		compile_first=config.compile_first,
	)


def encode_legal_actions(info: dict[str, Any], action_encoder: ActionTextEncoder) -> np.ndarray:
	return np.stack([action_encoder.encode(action) for action in info["legal_actions"]]).astype(np.float32)


def collect_expert_samples(
	env: TribesEnv,
	state_encoder: JsonStateEncoder,
	action_encoder: ActionTextEncoder,
	config: RankDistillConfig,
	iteration: int,
) -> list[RankSample]:
	samples: list[RankSample] = []
	obs, info = env.reset(seed=config.seed + iteration)
	state = state_encoder.encode(obs["state_json"])
	episode_steps = 0
	while len(samples) < config.collection_steps:
		action_features = encode_legal_actions(info, action_encoder)
		expert = env.bridge.expert_action(
			expert_type=config.expert_type,
			expert_seed=config.seed + iteration * 100_000 + len(samples),
			fm_calls=config.expert_fm_calls,
			rollout_length=config.expert_rollout_length,
			rollouts=config.expert_rollouts,
			time_millis=config.expert_time_millis,
		)
		action_index = int(expert["actionIndex"])
		if action_index >= action_features.shape[0]:
			raise RuntimeError(f"Expert action index {action_index} is outside legal action count {action_features.shape[0]}")
		samples.append(RankSample(state=state, action_features=action_features, expert_local_index=action_index))
		next_obs, _, terminated, truncated, next_info = env.step(action_index)
		episode_steps += 1
		if terminated or truncated or episode_steps >= config.max_steps_per_episode:
			obs, info = env.reset(seed=config.seed + iteration + len(samples))
			state = state_encoder.encode(obs["state_json"])
			episode_steps = 0
		else:
			state = state_encoder.encode(next_obs["state_json"])
			info = next_info
	return samples


def iter_minibatches(samples: list[RankSample], batch_size: int) -> list[list[RankSample]]:
	indices = np.arange(len(samples))
	np.random.shuffle(indices)
	return [[samples[int(index)] for index in indices[start : start + batch_size]] for start in range(0, len(indices), batch_size)]


def evaluate_agent(agent: RankAgent, config: RankDistillConfig, state_encoder: JsonStateEncoder, action_encoder: ActionTextEncoder) -> dict[str, Any]:
	env = make_env(config)
	rewards: list[float] = []
	for episode in range(config.eval_episodes):
		obs, info = env.reset(seed=config.seed + config.eval_seed_offset + episode)
		state = state_encoder.encode(obs["state_json"])
		total = 0.0
		steps = 0
		for _ in range(config.eval_max_steps_per_episode):
			action_features = encode_legal_actions(info, action_encoder)
			action = agent.select_action(state, action_features)
			next_obs, reward, terminated, truncated, next_info = env.step(action)
			total += float(reward)
			steps += 1
			if terminated or truncated:
				break
			state = state_encoder.encode(next_obs["state_json"])
			info = next_info
		rewards.append(total)
		LOGGER.info("eval episode=%d reward=%.3f steps=%d", episode + 1, total, steps)
	env.close()
	return {"episode_rewards": rewards, "mean_reward": float(np.mean(rewards)) if rewards else 0.0}


def run_training(config: RankDistillConfig) -> dict[str, Any]:
	random.seed(config.seed)
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	env = make_env(config)
	state_encoder = JsonStateEncoder(config.encoder_mode, config.engineered_dim, config.raw_dim)
	action_encoder = ActionTextEncoder(config.action_feature_dim)
	obs, _ = env.reset(seed=config.seed)
	state_dim = state_encoder.encode(obs["state_json"]).shape[0]
	path = Path(config.checkpoint_path)
	if config.resume and path.exists():
		agent, extra = RankAgent.load(path, state_dim, config)
		stats = extra.get("stats", {"loss": [], "accuracy": [], "eval_rewards": []})
	else:
		agent = RankAgent(state_dim, config)
		stats = {"loss": [], "accuracy": [], "eval_rewards": []}
	replay = RankBatch()
	for iteration in range(agent.iteration, config.total_iterations):
		new_samples = collect_expert_samples(env, state_encoder, action_encoder, config, iteration)
		replay.extend(new_samples, config.expert_replay_size)
		train_samples = replay.samples if config.expert_replay_size > 0 else new_samples
		losses: list[float] = []
		accuracies: list[float] = []
		for _ in range(config.train_epochs_per_collection):
			for minibatch in iter_minibatches(train_samples, config.batch_size):
				result = agent.train_batch(minibatch)
				losses.append(result["loss"])
				accuracies.append(result["accuracy"])
		agent.iteration = iteration + 1
		mean_loss = float(np.mean(losses)) if losses else 0.0
		mean_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
		stats["loss"].append(mean_loss)
		stats["accuracy"].append(mean_accuracy)
		eval_reward = None
		if config.eval_every_iterations > 0 and agent.iteration % config.eval_every_iterations == 0:
			results = evaluate_agent(agent, config, state_encoder, action_encoder)
			eval_reward = float(results["mean_reward"])
			stats["eval_rewards"].append(eval_reward)
			if eval_reward > agent.best_eval_reward:
				agent.best_eval_reward = eval_reward
				agent.save(path.with_suffix(".best.pt"), extra={"stats": stats})
		agent.save(path, extra={"stats": stats})
		LOGGER.info(
			"iter=%d samples=%d loss=%.5f accuracy=%.3f eval_reward=%s best_eval=%s",
			agent.iteration,
			len(train_samples),
			mean_loss,
			mean_accuracy,
			"none" if eval_reward is None else f"{eval_reward:.3f}",
			"none" if agent.best_eval_reward == float("-inf") else f"{agent.best_eval_reward:.3f}",
		)
	env.close()
	return stats


def run_evaluation(config: RankDistillConfig) -> dict[str, Any]:
	state_encoder = JsonStateEncoder(config.encoder_mode, config.engineered_dim, config.raw_dim)
	action_encoder = ActionTextEncoder(config.action_feature_dim)
	env = make_env(config)
	obs, _ = env.reset(seed=config.seed)
	state_dim = state_encoder.encode(obs["state_json"]).shape[0]
	env.close()
	agent, _ = RankAgent.load(config.checkpoint_path, state_dim, config)
	return evaluate_agent(agent, config, state_encoder, action_encoder)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Action-conditioned MCTS expert distillation for Tribes")
	subparsers = parser.add_subparsers(dest="command", required=True)

	def add_shared(subparser: argparse.ArgumentParser) -> None:
		subparser.add_argument("--level-file", default="tribes/levels/SampleLevel.csv")
		subparser.add_argument("--game-mode", default="SCORE")
		subparser.add_argument("--seed", type=int, default=123)
		subparser.add_argument("--checkpoint-path", default="checkpoints/expert_rank_distill.pt")
		subparser.add_argument("--encoder-mode", choices=("engineered", "raw", "combined"), default="engineered")
		subparser.add_argument("--engineered-dim", type=int, default=ENGINEERED_FEATURE_DIM)
		subparser.add_argument("--raw-dim", type=int, default=RAW_FEATURE_DIM)
		subparser.add_argument("--action-feature-dim", type=int, default=192)
		subparser.add_argument("--hidden-dim", type=int, default=256)
		subparser.add_argument("--compile-first", action=argparse.BooleanOptionalAction, default=True)
		subparser.add_argument("--device", default="cpu")

	train = subparsers.add_parser("train")
	add_shared(train)
	train.add_argument("--total-iterations", type=int, default=40)
	train.add_argument("--collection-steps", type=int, default=256)
	train.add_argument("--batch-size", type=int, default=64)
	train.add_argument("--train-epochs-per-collection", type=int, default=4)
	train.add_argument("--expert-replay-size", type=int, default=4096)
	train.add_argument("--learning-rate", type=float, default=3e-4)
	train.add_argument("--expert-type", choices=("MCTS", "pMCTS"), default="pMCTS")
	train.add_argument("--expert-fm-calls", type=int, default=250)
	train.add_argument("--expert-rollout-length", type=int, default=12)
	train.add_argument("--expert-rollouts", action=argparse.BooleanOptionalAction, default=False)
	train.add_argument("--expert-time-millis", type=int, default=10_000)
	train.add_argument("--eval-every-iterations", type=int, default=5)
	train.add_argument("--eval-episodes", type=int, default=5)
	train.add_argument("--eval-max-steps-per-episode", type=int, default=512)
	train.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)

	eval_parser = subparsers.add_parser("eval")
	add_shared(eval_parser)
	eval_parser.add_argument("--eval-episodes", type=int, default=5)
	eval_parser.add_argument("--eval-max-steps-per-episode", type=int, default=512)

	return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RankDistillConfig:
	config = RankDistillConfig(
		level_file=args.level_file,
		game_mode=args.game_mode,
		seed=args.seed,
		checkpoint_path=args.checkpoint_path,
		compile_first=args.compile_first,
		encoder_mode=args.encoder_mode,
		engineered_dim=args.engineered_dim,
		raw_dim=args.raw_dim,
		action_feature_dim=args.action_feature_dim,
		hidden_dim=args.hidden_dim,
		device=args.device,
	)
	if args.command == "train":
		config.total_iterations = args.total_iterations
		config.collection_steps = args.collection_steps
		config.batch_size = args.batch_size
		config.train_epochs_per_collection = args.train_epochs_per_collection
		config.expert_replay_size = args.expert_replay_size
		config.learning_rate = args.learning_rate
		config.expert_type = args.expert_type
		config.expert_fm_calls = args.expert_fm_calls
		config.expert_rollout_length = args.expert_rollout_length
		config.expert_rollouts = args.expert_rollouts
		config.expert_time_millis = args.expert_time_millis
		config.eval_every_iterations = args.eval_every_iterations
		config.eval_episodes = args.eval_episodes
		config.eval_max_steps_per_episode = args.eval_max_steps_per_episode
		config.resume = args.resume
	elif args.command == "eval":
		config.eval_episodes = args.eval_episodes
		config.eval_max_steps_per_episode = args.eval_max_steps_per_episode
	return config


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(message)s")
	args = parse_args()
	config = config_from_args(args)
	if args.command == "train":
		stats = run_training(config)
		print(json.dumps(stats, indent=2, sort_keys=True))
	else:
		results = run_evaluation(config)
		print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
	main()
