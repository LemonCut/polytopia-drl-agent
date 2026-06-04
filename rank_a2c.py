from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dqn import DQNAgent, DQNConfig, ENGINEERED_FEATURE_DIM, RAW_FEATURE_DIM, JsonStateEncoder
from expert_rank_distill import ActionTextEncoder, RankAgent, RankDistillConfig, RankPolicy, RankSample, collate_samples, encode_legal_actions
from tribes_env import TribesEnv


LOGGER = logging.getLogger(__name__)


@dataclass
class RankA2CConfig:
	level_file: str = "tribes/levels/SampleLevel.csv"
	game_mode: str = "SCORE"
	seed: int = 123
	compile_first: bool = True
	encoder_mode: str = "engineered"
	engineered_dim: int = ENGINEERED_FEATURE_DIM
	raw_dim: int = RAW_FEATURE_DIM
	action_feature_dim: int = 192
	hidden_dim: int = 256
	learning_rate: float = 1e-5
	actor_learning_rate: float | None = None
	critic_learning_rate: float | None = None
	gamma: float = 0.99
	gae_lambda: float = 0.95
	entropy_coef: float = 0.01
	value_coef: float = 0.5
	anchor_kl_coef: float = 0.0
	gradient_clip_norm: float = 1.0
	total_iterations: int = 20
	rollout_steps: int = 256
	update_epochs: int = 2
	critic_warmup_iterations: int = 0
	actor_update_scale: float = 1.0
	self_imitation_coef: float = 0.0
	self_imitation_buffer_size: int = 0
	self_imitation_batch_size: int = 64
	self_imitation_updates: int = 1
	self_imitation_min_advantage: float = 0.0
	self_imitation_max_weight: float = 5.0
	controlled_tribes: str = ""
	opponent_policy: str = "agent"
	opponent_dqn_checkpoint: str = "checkpoints/dqn.pt"
	max_steps_per_episode: int = 2_000
	reward_scale: float = 100.0
	eval_every_iterations: int = 5
	eval_episodes: int = 5
	eval_max_steps_per_episode: int = 512
	eval_seed_offset: int = 10_000
	init_rank_checkpoint: str = "checkpoints/expert_rank_distill.pt"
	init_from_rank: bool = True
	checkpoint_path: str = "checkpoints/rank_a2c.pt"
	resume: bool = False
	device: str = "cpu"


class RankActorCritic(nn.Module):
	def __init__(self, state_dim: int, action_feature_dim: int, hidden_dim: int) -> None:
		super().__init__()
		self.actor = RankPolicy(state_dim, action_feature_dim, hidden_dim)
		self.critic = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, 1),
		)

	def forward(self, states: torch.Tensor, action_features: torch.Tensor, action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		logits = self.actor(states, action_features, action_mask)
		values = self.critic(states).squeeze(-1)
		return logits, values


@dataclass
class RolloutStep:
	state: np.ndarray
	action_features: np.ndarray
	action: int
	reward: float
	done: bool
	log_prob: float
	value: float


@dataclass
class SelfImitationSample:
	step: RolloutStep
	weight: float


@dataclass
class DQNOpponent:
	agent: DQNAgent
	encoder: JsonStateEncoder

	def act(self, obs: dict[str, str], info: dict[str, Any]) -> int:
		state = self.encoder.encode(obs["state_json"])
		action_mask = np.asarray(info["action_mask"], dtype=np.bool_)
		return self.agent.select_action(state, action_mask, epsilon=0.0, greedy=True)


class RankA2CAgent:
	def __init__(self, state_dim: int, config: RankA2CConfig) -> None:
		self.state_dim = state_dim
		self.config = config
		self.device = torch.device(config.device)
		self.model = RankActorCritic(state_dim, config.action_feature_dim, config.hidden_dim).to(self.device)
		self.reference_actor: RankPolicy | None = None
		actor_lr = config.actor_learning_rate if config.actor_learning_rate is not None else config.learning_rate
		critic_lr = config.critic_learning_rate if config.critic_learning_rate is not None else config.learning_rate
		self.optimizer = torch.optim.AdamW(
			[
				{"params": self.model.actor.parameters(), "lr": actor_lr},
				{"params": self.model.critic.parameters(), "lr": critic_lr},
			]
		)
		self.iteration = 0
		self.best_eval_reward = float("-inf")

	def load_actor_from_rank_checkpoint(self, path: str | Path) -> None:
		rank_agent = self._load_rank_agent(path)
		self.model.actor.load_state_dict(rank_agent.model.state_dict())
		self.set_reference_from_rank_checkpoint(path)

	def set_reference_from_rank_checkpoint(self, path: str | Path) -> None:
		rank_agent = self._load_rank_agent(path)
		self.reference_actor = RankPolicy(self.state_dim, self.config.action_feature_dim, self.config.hidden_dim).to(self.device)
		self.reference_actor.load_state_dict(rank_agent.model.state_dict())
		self.reference_actor.eval()
		for param in self.reference_actor.parameters():
			param.requires_grad_(False)

	def _load_rank_agent(self, path: str | Path) -> RankAgent:
		rank_config = RankDistillConfig(
			encoder_mode=self.config.encoder_mode,
			engineered_dim=self.config.engineered_dim,
			raw_dim=self.config.raw_dim,
			action_feature_dim=self.config.action_feature_dim,
			hidden_dim=self.config.hidden_dim,
			device=self.config.device,
		)
		rank_agent, _ = RankAgent.load(path, self.state_dim, rank_config)
		return rank_agent

	@torch.no_grad()
	def act(self, state: np.ndarray, action_features: np.ndarray, greedy: bool = False) -> tuple[int, float, float]:
		self.model.eval()
		states, actions, mask, _ = collate_samples(
			[RankSample(state=state, action_features=action_features, expert_local_index=0)],
			self.device,
		)
		logits, values = self.model(states, actions, mask)
		dist = torch.distributions.Categorical(logits=logits)
		action = torch.argmax(logits, dim=1) if greedy else dist.sample()
		log_prob = dist.log_prob(action)
		return int(action.item()), float(log_prob.item()), float(values.item())

	def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(
			{
				"config": asdict(self.config),
				"state_dim": self.state_dim,
				"model_state_dict": self.model.state_dict(),
				"optimizer_state_dict": self.optimizer.state_dict(),
				"iteration": self.iteration,
				"best_eval_reward": self.best_eval_reward,
				"extra": extra or {},
			},
			path,
		)

	@classmethod
	def load(cls, path: str | Path, config: RankA2CConfig | None = None) -> tuple["RankA2CAgent", dict[str, Any]]:
		checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
		loaded_config = RankA2CConfig(**checkpoint["config"])
		if config is not None:
			for field in ("encoder_mode", "engineered_dim", "raw_dim", "action_feature_dim", "hidden_dim"):
				if getattr(config, field) != getattr(loaded_config, field):
					raise ValueError(f"Checkpoint config mismatch for {field}")
			loaded_config = config
		agent = cls(int(checkpoint["state_dim"]), loaded_config)
		agent.model.load_state_dict(checkpoint["model_state_dict"])
		try:
			agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		except ValueError:
			LOGGER.warning("Skipping optimizer state load because optimizer parameter groups changed")
		agent.iteration = int(checkpoint.get("iteration", 0))
		agent.best_eval_reward = float(checkpoint.get("best_eval_reward", float("-inf")))
		return agent, checkpoint.get("extra", {})

	def update(self, rollout: list[RolloutStep], last_value: float, *, actor_scale: float = 1.0) -> dict[str, float]:
		advantages, returns = compute_advantages_and_returns(rollout, last_value, self.config.gamma, self.config.gae_lambda)
		advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
		advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
		returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
		targets = torch.as_tensor([step.action for step in rollout], dtype=torch.long, device=self.device)

		stats: list[dict[str, float]] = []
		for _ in range(self.config.update_epochs):
			states, actions, mask, _ = collate_samples(
				[RankSample(state=step.state, action_features=step.action_features, expert_local_index=0) for step in rollout],
				self.device,
			)
			logits, values_t = self.model(states, actions, mask)
			dist = torch.distributions.Categorical(logits=logits)
			log_probs = dist.log_prob(targets)
			entropy = dist.entropy().mean()
			policy_loss = -(log_probs * advantages_t).mean() * actor_scale
			value_loss = torch.nn.functional.mse_loss(values_t, returns_t)
			anchor_kl = torch.zeros((), dtype=torch.float32, device=self.device)
			if self.config.anchor_kl_coef > 0 and self.reference_actor is not None:
				with torch.no_grad():
					ref_logits = self.reference_actor(states, actions, mask)
					ref_probs = torch.softmax(ref_logits, dim=-1)
				log_probs_current = torch.log_softmax(logits, dim=-1)
				ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
				anchor_kl = (ref_probs * (ref_log_probs - log_probs_current)).sum(dim=-1).mean()
			loss = (
				policy_loss
				+ self.config.value_coef * value_loss
				- self.config.entropy_coef * entropy
				+ self.config.anchor_kl_coef * anchor_kl
			)

			self.optimizer.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
			self.optimizer.step()
			stats.append(
				{
					"loss": float(loss.item()),
					"policy_loss": float(policy_loss.item()),
					"value_loss": float(value_loss.item()),
					"entropy": float(entropy.item()),
					"anchor_kl": float(anchor_kl.item()),
				}
			)
		return {key: float(np.mean([item[key] for item in stats])) for key in stats[0]}

	def self_imitation_update(self, samples: list[SelfImitationSample]) -> dict[str, float]:
		if not samples or self.config.self_imitation_coef <= 0:
			return {"self_imitation_loss": 0.0, "self_imitation_accuracy": 0.0}
		stats: list[dict[str, float]] = []
		for _ in range(self.config.self_imitation_updates):
			batch_size = min(self.config.self_imitation_batch_size, len(samples))
			batch_indices = np.random.choice(len(samples), size=batch_size, replace=False)
			batch = [samples[int(index)] for index in batch_indices]
			states, actions, mask, _ = collate_samples(
				[RankSample(state=item.step.state, action_features=item.step.action_features, expert_local_index=0) for item in batch],
				self.device,
			)
			targets = torch.as_tensor([item.step.action for item in batch], dtype=torch.long, device=self.device)
			weights = torch.as_tensor([item.weight for item in batch], dtype=torch.float32, device=self.device)
			weights = weights / (weights.mean() + 1e-8)
			logits, _ = self.model(states, actions, mask)
			per_item_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
			loss = (per_item_loss * weights).mean() * self.config.self_imitation_coef
			with torch.no_grad():
				accuracy = (torch.argmax(logits, dim=1) == targets).float().mean()
			self.optimizer.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.config.gradient_clip_norm)
			self.optimizer.step()
			stats.append({"self_imitation_loss": float(loss.item()), "self_imitation_accuracy": float(accuracy.item())})
		return {key: float(np.mean([item[key] for item in stats])) for key in stats[0]}


def compute_advantages_and_returns(rollout: list[RolloutStep], last_value: float, gamma: float, gae_lambda: float) -> tuple[list[float], list[float]]:
	rewards = [step.reward for step in rollout]
	dones = [step.done for step in rollout]
	values = [step.value for step in rollout] + [last_value]
	advantages: list[float] = []
	gae = 0.0
	for t in reversed(range(len(rollout))):
		delta = rewards[t] + gamma * values[t + 1] * (1.0 - float(dones[t])) - values[t]
		gae = delta + gamma * gae_lambda * (1.0 - float(dones[t])) * gae
		advantages.insert(0, gae)
	returns = [adv + value for adv, value in zip(advantages, values[:-1])]
	return advantages, returns


def make_env(config: RankA2CConfig) -> TribesEnv:
	return TribesEnv(
		level_file=config.level_file,
		game_mode=config.game_mode,
		seed=config.seed,
		max_episode_steps=config.max_steps_per_episode,
		compile_first=config.compile_first,
	)


def collect_rollout(
	env: TribesEnv,
	agent: RankA2CAgent,
	state_encoder: JsonStateEncoder,
	action_encoder: ActionTextEncoder,
	config: RankA2CConfig,
	iteration: int,
	dqn_opponent: DQNOpponent | None = None,
) -> tuple[list[RolloutStep], np.ndarray, np.ndarray]:
	obs, info = env.reset(seed=config.seed + iteration)
	state = state_encoder.encode(obs["state_json"])
	rollout: list[RolloutStep] = []
	controlled_tribes = parse_controlled_tribes(config.controlled_tribes)
	env_steps = 0
	while len(rollout) < config.rollout_steps:
		active_tribe = info.get("active_tribe_id")
		if controlled_tribes is not None and active_tribe not in controlled_tribes:
			action = select_opponent_action(obs, info, config.opponent_policy, dqn_opponent)
			next_obs, _, terminated, truncated, next_info = env.step(action)
			env_steps += 1
			if terminated or truncated:
				obs, info = env.reset(seed=config.seed + iteration + env_steps)
				state = state_encoder.encode(obs["state_json"])
			else:
				state = state_encoder.encode(next_obs["state_json"])
				info = next_info
			continue
		action_features = encode_legal_actions(info, action_encoder)
		action, log_prob, value = agent.act(state, action_features, greedy=False)
		next_obs, reward, terminated, truncated, next_info = env.step(action)
		env_steps += 1
		rollout.append(
			RolloutStep(
				state=state,
				action_features=action_features,
				action=action,
				reward=float(reward) / config.reward_scale,
				done=bool(terminated or truncated),
				log_prob=log_prob,
				value=value,
			)
		)
		if terminated or truncated:
			obs, info = env.reset(seed=config.seed + iteration + len(rollout))
			state = state_encoder.encode(obs["state_json"])
		else:
			state = state_encoder.encode(next_obs["state_json"])
			info = next_info
	return rollout, state, encode_legal_actions(info, action_encoder)


def parse_controlled_tribes(value: str) -> set[int] | None:
	if not value.strip():
		return None
	return {int(item.strip()) for item in value.split(",") if item.strip()}


def select_opponent_action(obs: dict[str, str], info: dict[str, Any], policy: str, dqn_opponent: DQNOpponent | None = None) -> int:
	valid_actions = np.flatnonzero(np.asarray(info["action_mask"], dtype=np.bool_))
	if valid_actions.size == 0:
		raise RuntimeError("No legal actions available for opponent")
	if policy == "dqn":
		if dqn_opponent is None:
			raise RuntimeError("opponent_policy='dqn' requires a loaded DQN opponent")
		return dqn_opponent.act(obs, info)
	if policy == "end_turn":
		for index in valid_actions:
			action = info["legal_actions"][int(index)]
			if action.get("type") == "END_TURN":
				return int(index)
	return int(np.random.choice(valid_actions))


def evaluate_agent(agent: RankA2CAgent, config: RankA2CConfig, state_encoder: JsonStateEncoder, action_encoder: ActionTextEncoder) -> dict[str, Any]:
	env = make_env(config)
	rewards: list[float] = []
	for episode in range(config.eval_episodes):
		obs, info = env.reset(seed=config.seed + config.eval_seed_offset + episode)
		state = state_encoder.encode(obs["state_json"])
		total = 0.0
		steps = 0
		for _ in range(config.eval_max_steps_per_episode):
			action_features = encode_legal_actions(info, action_encoder)
			action, _, _ = agent.act(state, action_features, greedy=True)
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


def initialize_agent(config: RankA2CConfig) -> tuple[RankA2CAgent, JsonStateEncoder, ActionTextEncoder]:
	state_encoder = JsonStateEncoder(config.encoder_mode, config.engineered_dim, config.raw_dim)
	action_encoder = ActionTextEncoder(config.action_feature_dim)
	env = make_env(config)
	obs, _ = env.reset(seed=config.seed)
	state_dim = state_encoder.encode(obs["state_json"]).shape[0]
	env.close()
	path = Path(config.checkpoint_path)
	if config.resume and path.exists():
		agent, _ = RankA2CAgent.load(path, config)
		if config.anchor_kl_coef > 0:
			agent.set_reference_from_rank_checkpoint(config.init_rank_checkpoint)
	else:
		agent = RankA2CAgent(state_dim, config)
		if config.init_from_rank:
			agent.load_actor_from_rank_checkpoint(config.init_rank_checkpoint)
	return agent, state_encoder, action_encoder


def load_dqn_opponent(config: RankA2CConfig) -> DQNOpponent | None:
	if config.opponent_policy != "dqn":
		return None
	dqn_config = DQNConfig(
		level_file=config.level_file,
		game_mode=config.game_mode,
		seed=config.seed,
		compile_first=False,
		encoder_mode="combined",
		engineered_dim=config.engineered_dim,
		raw_dim=config.raw_dim,
		hidden_dim=256,
		checkpoint_path=config.opponent_dqn_checkpoint,
		device=config.device,
	)
	agent, _ = DQNAgent.load(config.opponent_dqn_checkpoint, config=dqn_config)
	agent.online.eval()
	return DQNOpponent(
		agent=agent,
		encoder=JsonStateEncoder(dqn_config.encoder_mode, dqn_config.engineered_dim, dqn_config.raw_dim),
	)


def run_training(config: RankA2CConfig) -> dict[str, Any]:
	random.seed(config.seed)
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	agent, state_encoder, action_encoder = initialize_agent(config)
	dqn_opponent = load_dqn_opponent(config)
	env = make_env(config)
	stats: dict[str, list[float]] = {
		"loss": [],
		"policy_loss": [],
		"value_loss": [],
		"entropy": [],
		"anchor_kl": [],
		"self_imitation_loss": [],
		"self_imitation_accuracy": [],
		"eval_rewards": [],
	}
	self_imitation_buffer: list[SelfImitationSample] = []
	for iteration in range(agent.iteration, config.total_iterations):
		rollout, final_state, final_action_features = collect_rollout(
			env,
			agent,
			state_encoder,
			action_encoder,
			config,
			iteration,
			dqn_opponent,
		)
		_, _, last_value = agent.act(final_state, final_action_features, greedy=True)
		if config.self_imitation_buffer_size > 0 and config.self_imitation_coef > 0:
			advantages, _ = compute_advantages_and_returns(rollout, last_value, config.gamma, config.gae_lambda)
			for step, advantage in zip(rollout, advantages):
				if advantage > config.self_imitation_min_advantage:
					weight = min(config.self_imitation_max_weight, max(1e-3, float(advantage)))
					self_imitation_buffer.append(SelfImitationSample(step=step, weight=weight))
			if len(self_imitation_buffer) > config.self_imitation_buffer_size:
				del self_imitation_buffer[: len(self_imitation_buffer) - config.self_imitation_buffer_size]
		actor_scale = 0.0 if agent.iteration < config.critic_warmup_iterations else config.actor_update_scale
		result = agent.update(rollout, last_value, actor_scale=actor_scale)
		self_imitation_result = agent.self_imitation_update(self_imitation_buffer)
		agent.iteration = iteration + 1
		for key in ("loss", "policy_loss", "value_loss", "entropy", "anchor_kl"):
			stats[key].append(result[key])
		for key in ("self_imitation_loss", "self_imitation_accuracy"):
			stats[key].append(self_imitation_result[key])
		eval_reward = None
		if config.eval_every_iterations > 0 and agent.iteration % config.eval_every_iterations == 0:
			eval_result = evaluate_agent(agent, config, state_encoder, action_encoder)
			eval_reward = eval_result["mean_reward"]
			stats["eval_rewards"].append(float(eval_reward))
			if eval_reward > agent.best_eval_reward:
				agent.best_eval_reward = float(eval_reward)
				agent.save(Path(config.checkpoint_path).with_suffix(".best.pt"), extra={"stats": stats})
		agent.save(config.checkpoint_path, extra={"stats": stats})
		LOGGER.info(
			"iter=%d loss=%.4f policy=%.4f value=%.4f entropy=%.4f anchor_kl=%.4f sil=%.4f sil_acc=%.3f eval=%s best=%s",
			agent.iteration,
			result["loss"],
			result["policy_loss"],
			result["value_loss"],
			result["entropy"],
			result["anchor_kl"],
			self_imitation_result["self_imitation_loss"],
			self_imitation_result["self_imitation_accuracy"],
			"none" if eval_reward is None else f"{eval_reward:.1f}",
			"none" if agent.best_eval_reward == float("-inf") else f"{agent.best_eval_reward:.1f}",
		)
	env.close()
	return stats


def run_evaluation(config: RankA2CConfig) -> dict[str, Any]:
	agent, state_encoder, action_encoder = initialize_agent(config)
	return evaluate_agent(agent, config, state_encoder, action_encoder)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Non-PPO action-conditioned A2C fine-tuning for Tribes")
	subparsers = parser.add_subparsers(dest="command", required=True)

	def add_shared(subparser: argparse.ArgumentParser) -> None:
		subparser.add_argument("--level-file", default="tribes/levels/SampleLevel.csv")
		subparser.add_argument("--game-mode", default="SCORE")
		subparser.add_argument("--seed", type=int, default=123)
		subparser.add_argument("--checkpoint-path", default="checkpoints/rank_a2c.pt")
		subparser.add_argument("--init-rank-checkpoint", default="checkpoints/expert_rank_distill.pt")
		subparser.add_argument("--encoder-mode", choices=("engineered", "raw", "combined"), default="engineered")
		subparser.add_argument("--engineered-dim", type=int, default=ENGINEERED_FEATURE_DIM)
		subparser.add_argument("--raw-dim", type=int, default=RAW_FEATURE_DIM)
		subparser.add_argument("--action-feature-dim", type=int, default=192)
		subparser.add_argument("--hidden-dim", type=int, default=256)
		subparser.add_argument("--compile-first", action=argparse.BooleanOptionalAction, default=True)
		subparser.add_argument("--device", default="cpu")

	train = subparsers.add_parser("train")
	add_shared(train)
	train.add_argument("--total-iterations", type=int, default=20)
	train.add_argument("--rollout-steps", type=int, default=256)
	train.add_argument("--update-epochs", type=int, default=2)
	train.add_argument("--learning-rate", type=float, default=1e-5)
	train.add_argument("--actor-learning-rate", type=float, default=None)
	train.add_argument("--critic-learning-rate", type=float, default=None)
	train.add_argument("--gamma", type=float, default=0.99)
	train.add_argument("--gae-lambda", type=float, default=0.95)
	train.add_argument("--entropy-coef", type=float, default=0.01)
	train.add_argument("--value-coef", type=float, default=0.5)
	train.add_argument("--anchor-kl-coef", type=float, default=0.0)
	train.add_argument("--reward-scale", type=float, default=100.0)
	train.add_argument("--critic-warmup-iterations", type=int, default=0)
	train.add_argument("--actor-update-scale", type=float, default=1.0)
	train.add_argument("--self-imitation-coef", type=float, default=0.0)
	train.add_argument("--self-imitation-buffer-size", type=int, default=0)
	train.add_argument("--self-imitation-batch-size", type=int, default=64)
	train.add_argument("--self-imitation-updates", type=int, default=1)
	train.add_argument("--self-imitation-min-advantage", type=float, default=0.0)
	train.add_argument("--self-imitation-max-weight", type=float, default=5.0)
	train.add_argument("--controlled-tribes", default="")
	train.add_argument("--opponent-policy", choices=("random", "end_turn", "agent", "dqn"), default="agent")
	train.add_argument("--opponent-dqn-checkpoint", default="checkpoints/dqn.pt")
	train.add_argument("--eval-every-iterations", type=int, default=5)
	train.add_argument("--eval-episodes", type=int, default=5)
	train.add_argument("--eval-max-steps-per-episode", type=int, default=512)
	train.add_argument("--init-rank", action=argparse.BooleanOptionalAction, default=True)
	train.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)

	eval_parser = subparsers.add_parser("eval")
	add_shared(eval_parser)
	eval_parser.add_argument("--eval-episodes", type=int, default=5)
	eval_parser.add_argument("--eval-max-steps-per-episode", type=int, default=512)
	eval_parser.add_argument("--init-rank", action=argparse.BooleanOptionalAction, default=True)
	eval_parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
	return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RankA2CConfig:
	config = RankA2CConfig(
		level_file=args.level_file,
		game_mode=args.game_mode,
		seed=args.seed,
		checkpoint_path=args.checkpoint_path,
		init_rank_checkpoint=args.init_rank_checkpoint,
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
		config.rollout_steps = args.rollout_steps
		config.update_epochs = args.update_epochs
		config.learning_rate = args.learning_rate
		config.actor_learning_rate = args.actor_learning_rate
		config.critic_learning_rate = args.critic_learning_rate
		config.gamma = args.gamma
		config.gae_lambda = args.gae_lambda
		config.entropy_coef = args.entropy_coef
		config.value_coef = args.value_coef
		config.anchor_kl_coef = args.anchor_kl_coef
		config.reward_scale = args.reward_scale
		config.critic_warmup_iterations = args.critic_warmup_iterations
		config.actor_update_scale = args.actor_update_scale
		config.self_imitation_coef = args.self_imitation_coef
		config.self_imitation_buffer_size = args.self_imitation_buffer_size
		config.self_imitation_batch_size = args.self_imitation_batch_size
		config.self_imitation_updates = args.self_imitation_updates
		config.self_imitation_min_advantage = args.self_imitation_min_advantage
		config.self_imitation_max_weight = args.self_imitation_max_weight
		config.controlled_tribes = args.controlled_tribes
		config.opponent_policy = args.opponent_policy
		config.opponent_dqn_checkpoint = args.opponent_dqn_checkpoint
		config.eval_every_iterations = args.eval_every_iterations
		config.eval_episodes = args.eval_episodes
		config.eval_max_steps_per_episode = args.eval_max_steps_per_episode
		config.init_from_rank = args.init_rank
		config.resume = args.resume
	else:
		config.eval_episodes = args.eval_episodes
		config.eval_max_steps_per_episode = args.eval_max_steps_per_episode
		config.init_from_rank = args.init_rank
		config.resume = args.resume
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
