from logging import ERROR
from typing import List, Tuple
import os

# import d4rl
import gymnasium as gym
from gym.spaces import Box, Space
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import pickle

from memory import ReplayMemory

# gym.logger.set_level(ERROR)  # Ignore warnings from Gym logger


ENVS = ['ant', 'halfcheetah', 'hopper', 'walker2d']  # Supported envs


class D4RLEnv():
  def __init__(self, env_name: str, absorbing: bool, load_data: bool=False):
    # self.env = gym.make(f'{env_name}-expert-v2')
    self.env = gym.make(f'{env_name}')
    self.env = self.env.unwrapped
    # if load_data: 
      # self.dataset = self.get_dataset()  # Load dataset before (potentially) adjusting observation_space (fails assertion check otherwise)
    self.env.action_space.high, self.env.action_space.low = torch.as_tensor(self.env.action_space.high), torch.as_tensor(self.env.action_space.low)  # Convert action space for action clipping

    self.absorbing = absorbing
    if absorbing: 
      self.env.observation_space = Box(low=np.concatenate([self.env.observation_space.low, np.zeros(1)]), high=np.concatenate([self.env.observation_space.high, np.ones(1)]))  # Append absorbing indicator bit to state dimension (assumes 1D state space)

  def reset(self) -> Tensor:
    state = self.env.reset()
    state = torch.tensor(state, dtype=torch.float32)#.unsqueeze(dim=0)  # Add batch dimension to state
    if self.absorbing: 
      state = torch.cat([state, torch.zeros(state.size(0), 1)], dim=1)  # Add absorbing indicator (zero) to state
    return state 

  def step(self, action: Tensor) -> Tuple[Tensor, float, bool]:
    action = action.clamp(min=self.env.action_space.low, max=self.env.action_space.high)  # Clip actions
    state, reward, terminated, truncation, _ = self.env.step(action.detach().numpy()) #self.env.step(action[0].detach().numpy())  # Remove batch dimension from action
    state = torch.tensor(state, dtype=torch.float32)#.unsqueeze(dim=0)  # Add batch dimension to state
    terminated = torch.tensor(terminated, dtype=torch.int32)
    if self.absorbing: 
      state = torch.cat([state, torch.zeros(state.size(0), 1)], dim=1)  # Add absorbing indicator (zero) to state (absorbing state rewriting done in replay memory)
    return state, reward, terminated, truncation

  def seed(self, seed: int) -> List[int]:
    return self.env.seed(seed)

  def render(self):
    return self.env.render()

  def close(self):
    self.env.close()

  @property
  def observation_space(self) -> Space:
    return self.env.observation_space

  @property
  def action_space(self) -> Space:
    return self.env.action_space

  @property
  def max_episode_steps(self) -> int:
    return self.env.max_episode_steps
  
  def get_demo(self, demo_path, verbose=0):
    """
    Extract demo from pickled file
    """
    self.dataset = {}
    self.dataset["observations"] = []
    self.dataset["next_observations"] = []
    self.dataset["actions"] = []
    self.dataset["terminals"] = []
    self.dataset["timeouts"] = []

    assert os.path.isfile(demo_path)

    with open(demo_path, "rb") as f:
      demo = pickle.load(f)
      print("demo.keys() = ", demo.keys())

    for obs, full_state, action in zip(demo["observations"], demo["full_states"], demo["actions"]):
      self.dataset["observations"].append(obs)
      self.dataset["actions"].append(action)
      self.dataset["terminals"].append(0)
      self.dataset["timeouts"].append(0)

    self.dataset["next_observations"] = self.dataset["observations"][1:]
    self.dataset["observations"] = self.dataset["observations"][:-1]
    self.dataset["actions"] = self.dataset["actions"][:-1]
    self.dataset["terminals"] = self.dataset["terminals"][:-1]
    self.dataset["terminals"][-1]= 1
    self.dataset["timeouts"] = self.dataset["timeouts"][:-1]

    return 

  def get_dataset(self, trajectories: int=0, subsample: int=1) -> ReplayMemory:

    # Extract data
    states = torch.as_tensor(self.dataset['observations'], dtype=torch.float32)
    actions = torch.as_tensor(self.dataset['actions'], dtype=torch.float32)
    next_states = torch.as_tensor(self.dataset['next_observations'], dtype=torch.float32)
    terminals = torch.as_tensor(self.dataset['terminals'], dtype=torch.float32)
    timeouts = torch.as_tensor(self.dataset['timeouts'], dtype=torch.float32)
    state_size, action_size = states.size(1), actions.size(1)
    # Split into separate trajectories
    states_list, actions_list, next_states_list, terminals_list, weights_list, timeouts_list = [], [], [], [], [], []
    terminal_idxs, timeout_idxs = terminals.nonzero().flatten(), timeouts.nonzero().flatten()
    ep_end_idxs = torch.sort(torch.cat([torch.tensor([-1]), terminal_idxs, timeout_idxs], dim=0))[0]
    for i in range(len(ep_end_idxs) - 1):
      states_list.append(states[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])
      actions_list.append(actions[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])
      next_states_list.append(next_states[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])
      terminals_list.append(terminals[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])  # Only store true terminations; timeouts should not be treated as such
      timeouts_list.append(timeouts[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])  # Store if episode terminated due to timeout
      weights_list.append(torch.ones_like(terminals_list[-1]))  # Add an importance weight of 1 to every transition
    # Pick number of trajectories
    if trajectories > 0:
      states_list = states_list[:trajectories]
      actions_list = actions_list[:trajectories]
      next_states_list = next_states_list[:trajectories]
      terminals_list = terminals_list[:trajectories]
      timeouts_list = timeouts_list[:trajectories]
      weights_list = weights_list[:trajectories]
    num_trajectories = len(states_list)
    # Wrap for absorbing states
    if self.absorbing:
      absorbing_state, absorbing_action = torch.cat([torch.zeros(1, state_size), torch.ones(1, 1)], dim=1), torch.zeros(1, action_size)  # Create absorbing state and absorbing action
      for i in range(len(states_list)):
        # Append absorbing indicator (zero)
        states_list[i] = torch.cat([states_list[i], torch.zeros(states_list[i].size(0), 1)], dim=1)
        next_states_list[i] = torch.cat([next_states_list[i], torch.zeros(next_states_list[i].size(0), 1)], dim=1)
        if not timeouts_list[i][-1]:  # Apply for episodes that did not terminate due to time limits
          # Replace the final next state with the absorbing state and overwrite terminal status
          next_states_list[i][-1] = absorbing_state
          terminals_list[i][-1] = 0
          weights_list[i][-1] = 1 / subsample  # Importance weight absorbing state as kept during subsampling
          # Add absorbing state to absorbing state transition
          states_list[i] = torch.cat([states_list[i], absorbing_state], dim=0)
          actions_list[i] = torch.cat([actions_list[i], absorbing_action], dim=0)
          next_states_list[i] = torch.cat([next_states_list[i], absorbing_state], dim=0)
          terminals_list[i] = torch.cat([terminals_list[i], torch.zeros(1)], dim=0)
          timeouts_list[i] = torch.cat([timeouts_list[i], torch.zeros(1)], dim=0)
          weights_list[i] = torch.cat([weights_list[i], torch.full((1, ), 1 / subsample)], dim=0)  # Importance weight absorbing state as kept during subsampling
    # Subsample within trajectories
    if subsample > 1:
      for i in range(len(states_list)):
        rand_start_idx, T = np.random.choice(subsample), len(states_list[i])  # Subsample from random index in 0 to N-1 (procedure from original GAIL implementation)
        idxs = range(rand_start_idx, T, subsample)
        if self.absorbing: idxs = sorted(list(set(idxs) | set([T - 2, T - 1])))  # Subsample but keep absorbing state transitions
        states_list[i] = states_list[i][idxs]
        actions_list[i] = actions_list[i][idxs]
        next_states_list[i] = next_states_list[i][idxs]
        terminals_list[i] = terminals_list[i][idxs]
        timeouts_list[i] = timeouts_list[i][idxs]
        weights_list[i] = weights_list[i][idxs]

    transitions = {'states': torch.cat(states_list, dim=0), 'actions': torch.cat(actions_list, dim=0), 'next_states': torch.cat(next_states_list, dim=0), 'terminals': torch.cat(terminals_list, dim=0), 'timeouts': torch.cat(timeouts_list, dim=0), 'weights': torch.cat(weights_list, dim=0), 'num_trajectories': num_trajectories}
    transitions['rewards'] = torch.zeros_like(transitions['terminals'])  # Pass 0 rewards to replay memory for interoperability/make sure reward information is not leaked to IL algorithm when data comes from an offline RL dataset
    return ReplayMemory(transitions['states'].size(0), state_size + (1 if self.absorbing else 0), action_size, self.absorbing, transitions=transitions)
