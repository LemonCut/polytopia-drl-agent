import argparse
import json
from pathlib import Path
import numpy as np
import random

from dqn import DQNAgent
from tribes_env import TribesEnv
from encoder import JsonStateEncoder

def run_game(agents, tribes, level_seed=-1, game_seed=-1, max_steps=2000, visuals=False, delay=0.0, compile_first=False):
    if len(agents) != len(tribes):
        raise ValueError("Number of agents must match number of tribes")
        
    python_agents = {}
    python_encoders = {}
    java_agent_names = []
    
    for i, agent_arg in enumerate(agents):
        path = Path(agent_arg)
        if path.exists() and path.suffix == ".pt":
            import torch
            from dqn import DQNConfig
            checkpoint = torch.load(path, map_location="cpu")
            loaded_config = DQNConfig(**checkpoint["config"])
            loaded_config.device = "cuda" if torch.cuda.is_available() else "cpu"
            agent, extra = DQNAgent.load(path, config=loaded_config)
            agent.online.eval()
            python_agents[i] = agent
            encoder = JsonStateEncoder(
                agent.config.encoder_mode, 
                agent.config.engineered_dim, 
                agent.config.raw_dim
            )
            python_encoders[i] = encoder
            java_agent_names.append("Python")
        else:
            java_agent_names.append(agent_arg)
            
    if level_seed == -1:
        level_seed = random.randint(1, 100000)
    if game_seed == -1:
        game_seed = random.randint(1, 100000)
        
    env = TribesEnv(
        level_seed=level_seed,
        tribes=tribes,
        seed=game_seed,
        max_episode_steps=max_steps,
        compile_first=compile_first
    )
    
    obs, info = env.reset(options={"agents": java_agent_names, "visuals": visuals})
    
    step_count = 0
    while True:
        active_tribe_id = info.get("active_tribe_id")
        if active_tribe_id is None:
            break
            
        if active_tribe_id in python_agents:
            agent = python_agents[active_tribe_id]
            encoder = python_encoders[active_tribe_id]
            state = encoder.encode(obs["state_json"])
            action_mask = np.asarray(info["action_mask"], dtype=np.bool_)
            action = agent.select_action(state, action_mask, epsilon=0.0, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
        else:
            obs, reward, terminated, truncated, info = env.agent_step()
            
        step_count += 1
        if terminated or truncated:
            break
            
        if delay > 0:
            import time
            time.sleep(delay)
            
    # Parse final state to report results
    state = json.loads(obs["state_json"])
    
    game_results = {
        "step_count": step_count,
        "players": []
    }
    
    tribes_state = state.get("tribes", {})
    for i, tribe_name in enumerate(tribes):
        tribe_data = tribes_state.get(str(i), {})
        score = tribe_data.get("score", 0)
        winner_status = tribe_data.get("winner", -1)
        status_str = "INCOMPLETE"
        if winner_status == 0:
            status_str = "WIN"
        elif winner_status == 1:
            status_str = "LOSS"
            
        agent_desc = agents[i]
        game_results["players"].append({
            "agent": agent_desc,
            "tribe": tribe_name,
            "score": score,
            "status": status_str
        })
        
    env.close()
    return game_results


def main():
    parser = argparse.ArgumentParser(description="Pit multiple agents against each other in Tribes")
    parser.add_argument("--agents", nargs="+", required=True, help="List of agent types (e.g. MCTS, Random) or paths to .pt DQN models")
    parser.add_argument("--tribes", nargs="+", required=True, help="List of tribes for each agent (e.g. 'Xin Xi' 'Imperius')")
    parser.add_argument("--level-seed", type=int, default=12345, help="Seed for the level generation")
    parser.add_argument("--game-seed", type=int, default=42, help="Seed for the game and agents")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum number of steps")
    parser.add_argument("--visuals", action="store_true", help="Enable live Java visuals")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay in seconds between moves (useful for slowing down visuals)")
    
    args = parser.parse_args()
    
    print(f"Starting game with tribes: {args.tribes}")
    print(f"Agents: {args.agents}\n")
    
    results = run_game(
        agents=args.agents,
        tribes=args.tribes,
        level_seed=args.level_seed,
        game_seed=args.game_seed,
        max_steps=args.max_steps,
        visuals=args.visuals,
        delay=args.delay,
        compile_first=True
    )
    
    print(f"\nGame Over after {results['step_count']} environment steps.")
    print("Scores:")
    for i, p in enumerate(results['players']):
        print(f"Player {i} ({p['tribe']} - {p['agent']}): {p['score']} points [{p['status']}]")

if __name__ == "__main__":
    main()
