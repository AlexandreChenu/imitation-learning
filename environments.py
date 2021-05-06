from logging import ERROR

import d4rl_pybullet
import gym
import numpy as np
import torch

from training import TransitionDataset

gym.logger.set_level(ERROR)  # Ignore warnings from Gym logger

D4RL_ENV_NAMES = ['ant-bullet-medium-v0', 'halfcheetah-bullet-medium-v0', 'hopper-bullet-medium-v0', 'walker2d-bullet-medium-v0']


# Test environment for testing the code
class PendulumEnv():
  def __init__(self, env_name=''):
    self.env = gym.make('Pendulum-v0')

  def reset(self):
    state = self.env.reset()
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state

  def step(self, action):
    state, reward, terminal, _ = self.env.step(action[0].detach().numpy())  # Remove batch dimension from action
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0), reward, terminal  # Add batch dimension to state

  def seed(self, seed):
    return self.env.seed(seed)

  def render(self):
    return self.env.render()

  def close(self):
    self.env.close()

  @property
  def observation_space(self):
    return self.env.observation_space

  @property
  def action_space(self):
    return self.env.action_space

  def get_dataset(self, size=0, dtype=torch.float):
    raise NotImplemented()



class D4RLEnv():
  def __init__(self, env_name):
    assert env_name in D4RL_ENV_NAMES
    
    self.env = gym.make(env_name)
    min_action_range, max_action_range = self.env.action_space.low[0], self.env.action_space.high[0]
    self.action_scale, self.action_loc = (max_action_range - min_action_range) / 2, (max_action_range + min_action_range) / 2

  def reset(self):
    state = self.env.reset()
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state

  def step(self, action):
    scaled_action = self.action_scale * action[0].detach().numpy() + self.action_loc # Assumes action is in (-1, 1) and rescales it accordingly; remove batch dimension from action
    state, reward, terminal, _ = self.env.step(scaled_action)
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0), reward, terminal  # Add batch dimension to state

  def seed(self, seed):
    return self.env.seed(seed)

  def render(self):
    return self.env.render()

  def close(self):
    self.env.close()

  @property
  def observation_space(self):
    return self.env.observation_space

  @property
  def action_space(self):
    return self.env.action_space

  def get_dataset(self, size=0, subsample=20):
    dataset = self.env.get_dataset()
    N = dataset['rewards'].shape[0]
    dataset_out = {'states': torch.as_tensor(dataset['observations'][:-1], dtype=torch.float32),
                   'actions': torch.as_tensor(dataset['actions'][:-1], dtype=torch.float32),
                   'rewards': torch.as_tensor(dataset['rewards'][:-1], dtype=torch.float32),
                   'next_states': torch.as_tensor(dataset['observations'][1:], dtype=torch.float32), 
                   'terminals': torch.as_tensor(dataset['terminals'][:-1], dtype=torch.float32)}
    # Postprocess
    if size > 0 and size < N:
      for key in dataset_out.keys():
        dataset_out[key] = dataset_out[key][0:size]
    if subsample > 0:
      for key in dataset_out.keys():
        dataset_out[key] = dataset_out[key][0::subsample]

    return TransitionDataset(dataset_out)


ENVS = {'ant': D4RLEnv, 'halfcheetah': D4RLEnv, 'hopper': D4RLEnv, 'pendulum': PendulumEnv, 'walker2d': D4RLEnv}