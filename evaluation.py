from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor

from environments import D4RLEnv
from models import SoftActor


# Evaluate agent with deterministic policy π
def evaluate_agent(actor: SoftActor, env: D4RLEnv, num_episodes: int, return_trajectories: bool=False, render: bool=False) -> Union[Tuple[List[List[float]], Dict[str, Tensor]], List[List[float]]]:
  returns, trajectories = [], []
  if render: env.render()  # PyBullet requires creating render window before first env reset, and then updates without requiring first call

  with torch.inference_mode():
    for _ in range(num_episodes):
      states, actions, rewards = [], [], []
      state, terminal, truncated = env.reset(), False, False
      while not (terminal or truncated):
          action = actor.get_greedy_action(state)  # Take greedy action
          next_state, reward, terminal, truncated = env.step(action)

          if return_trajectories:
            states.append(state)
            actions.append(action)
          rewards.append(reward)
          state = next_state
      returns.append(sum(rewards).item())

      if return_trajectories:
        # Collect trajectory data (including terminal signal, which may be needed for offline learning)
        terminals = torch.cat([torch.zeros(len(rewards) - 1), torch.ones(1)])
        trajectories.append({'states': torch.cat(states), 'actions': torch.cat(actions), 'rewards': torch.tensor(rewards, dtype=torch.float32), 'terminals': terminals})

  return (returns, trajectories) if return_trajectories else returns

# Evaluate agent with deterministic policy π
def evaluate_agent_dubins(actor: SoftActor, env: D4RLEnv, num_episodes: int, return_trajectories: bool=False, render: bool=False) -> Union[Tuple[List[List[float]], Dict[str, Tensor]], List[List[float]]]:
  returns, trajectories = [], []
  if render: env.render()  # PyBullet requires creating render window before first env reset, and then updates without requiring first call

  max_zone = 1

  with torch.inference_mode():
    for _ in range(num_episodes):
      states, actions, rewards = [], [], []
      state, terminal, truncated = env.reset(), False, False
      while not (terminal or truncated):
          action = actor.get_greedy_action(state)  # Take greedy action
          next_state, reward, terminal, truncated = env.step(action)

          if return_trajectories:
            states.append(state)
            actions.append(action)
          rewards.append(reward)
          state = next_state
		  
          zone = eval_zone(state[0][:2])
          if zone > max_zone: max_zone = zone
		  
      returns.append(sum(rewards).item())

      if return_trajectories:
        # Collect trajectory data (including terminal signal, which may be needed for offline learning)
        terminals = torch.cat([torch.zeros(len(rewards) - 1), torch.ones(1)])
        trajectories.append({'states': torch.cat(states), 'actions': torch.cat(actions), 'rewards': torch.tensor(rewards, dtype=torch.float32), 'terminals': terminals})

  return (returns, trajectories, max_zone) if return_trajectories else returns

def eval_zone(state):
	x = state[0]
	y = state[1]
	if y < 1.:
		if x < 1.:
			return 1
		elif  x < 2.:
			return 2
		elif  x < 3.:
			return 3
		elif  x < 4.:
			return 4
		else:
			return 5
	elif y < 2.:
		if  x > 4.:
			return 6
		elif  x > 3.:
			return 7
		elif x > 2.:
			return 8
		else:
			return 11
	elif y < 3.:
		if x < 1.:
			return 11
		elif x < 2.:
			return 10
		elif x < 3.:
			return 9
		elif x < 4.:
			return 20
		else :
			return 21

	elif y < 4.:
		if x < 1.:
			return 12
		elif x < 2.:
			return 15
		elif x < 3.:
			return 16
		elif x < 4:
			return 19
		else :
			return 22
	else:
		if x < 1.:
			return 13
		elif x < 2.:
			return 14
		elif x < 3.:
			return 17
		elif x < 4:
			return 18
		else :
			return 23