import numpy as np
import torch
from torch import autograd
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from models import update_target_network


# Replay memory returns transition tuples of the form (s, a, r, s', terminal)
class ReplayMemory(Dataset):
  def __init__(self, size, state_size, action_size, transitions=None):
    super().__init__()
    self.idx = 0
    self.size, self.full = size, False
    self.states, self.actions, self.rewards, self.next_states, self.terminals = torch.empty(size, state_size), torch.empty(size, action_size), torch.empty(size), torch.empty(size, state_size), torch.empty(size)
    if transitions is not None:
      trans_size = min(transitions['states'].size(0), size)  # Take data up to size of replay
      self.states[:trans_size], self.actions[:trans_size], self.rewards[:trans_size], self.next_states[:trans_size], self.terminals[:trans_size] = transitions['states'], transitions['actions'], transitions['rewards'], transitions['next_states'], transitions['terminals']
      self.idx = trans_size % self.size
      self.full = self.idx == 0 and trans_size > 0  # Replay is full if index has wrapped around (but not if there was no data)

  # Allows string-based access for entire data of one type, or int-based access for single transition
  def __getitem__(self, idx):
    if isinstance(idx, str):
      if idx == 'states':
        return self.states
      elif idx == 'actions':
        return self.actions
      elif idx == 'terminals':
        return self.terminals
    else:
      return dict(states=self.states[idx], actions=self.actions[idx], rewards=self.rewards[idx], next_states=self.next_states[idx], terminals=self.terminals[idx])

  def __len__(self):
    return self.terminals.size(0)

  def append(self, state, action, reward, next_state, terminal):
    self.states[self.idx], self.actions[self.idx], self.rewards[self.idx], self.next_states[self.idx], self.terminals[self.idx] = state, action, reward, next_state, terminal
    self.idx = (self.idx + 1) % self.size
    self.full = self.full or self.idx == 0

  # Returns a uniformly sampled valid transition index
  def _sample_idx(self):
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.size if self.full else self.idx - 1)
      valid_idx = idx != (self.idx - 1) % self.size  # Make sure data does not cross the memory index
    return idx

  def sample(self, n):
    idxs = [self._sample_idx() for _ in range(n)]
    transitions = [self[idx] for idx in idxs]
    return dict(states=torch.stack([t['states'] for t in transitions]), actions=torch.stack([t['actions'] for t in transitions]), rewards=torch.stack([t['rewards'] for t in transitions]), next_states=torch.stack([t['next_states'] for t in transitions]), terminals=torch.stack([t['terminals'] for t in transitions]))  # Note that stack creates new memory so SQIL does not overwrite original data

  def wrap_for_absorbing_states(self):  # TODO: Apply only if terminal state was not caused by a time limit? https://github.com/google-research/google-research/blob/master/dac/replay_buffer.py#L108
    absorbing_state = torch.cat([torch.zeros(self.states.size(1) - 1), torch.ones(1)], dim=0)
    self.next_states[(self.idx - 1) % self.size], self.terminals[(self.idx - 1) % self.size] = absorbing_state, False  # Replace terminal state with absorbing state and remove terminal
    self.append(absorbing_state, torch.zeros(self.actions.size(1)), 0, absorbing_state, False)  # Add absorbing state pair as next transition


# Performs one SAC update
def sac_update(actor, critic, log_alpha, target_critic, transitions, actor_optimiser, critic_optimiser, temperature_optimiser, discount, entropy_target, polyak_factor, max_grad_norm=0):
  states, actions, rewards, next_states, terminals = transitions['states'], transitions['actions'], transitions['rewards'], transitions['next_states'], transitions['terminals']
  alpha = log_alpha.exp()
  
  # Compute value function loss
  with torch.no_grad():
    new_next_policies = actor(next_states)
    new_next_actions = new_next_policies.sample()
    new_next_log_probs = new_next_policies.log_prob(new_next_actions)  # TODO: Deal with absorbing state? https://github.com/google-research/google-research/blob/master/dac/ddpg_td3.py#L146
    target_values = torch.min(*target_critic(next_states, new_next_actions)) - alpha * new_next_log_probs
    target_values = rewards + (1 - terminals) * discount * target_values
  values_1, values_2 = critic(states, actions)
  value_loss = F.mse_loss(values_1, target_values) + F.mse_loss(values_2, target_values)
  # Update critic
  critic_optimiser.zero_grad(set_to_none=True)
  value_loss.backward()
  if max_grad_norm > 0: clip_grad_norm_(critic.parameters(), max_grad_norm)
  critic_optimiser.step()

  # TODO: Remove absorbing states so actor/temperature are not updated on these
  # Compute policy loss
  new_policies = actor(states)
  new_actions = new_policies.rsample()
  new_log_probs = new_policies.log_prob(new_actions)
  new_values = torch.min(*critic(states, new_actions))
  policy_loss = (alpha.detach() * new_log_probs - new_values).mean()
  # Update actor
  actor_optimiser.zero_grad(set_to_none=True)
  policy_loss.backward()
  if max_grad_norm > 0: clip_grad_norm_(actor.parameters(), max_grad_norm)
  actor_optimiser.step()  

  # Compute temperature loss
  temperature_loss = -(alpha * (new_log_probs.detach() + entropy_target)).mean()
  # Update temperature
  temperature_optimiser.zero_grad(set_to_none=True)
  temperature_loss.backward()
  if max_grad_norm > 0: clip_grad_norm_(log_alpha, max_grad_norm)
  temperature_optimiser.step()

  # Update target critic
  update_target_network(critic, target_critic, polyak_factor)


# Performs a behavioural cloning update
def behavioural_cloning_update(actor, expert_trajectories, actor_optimiser, batch_size, max_grad_norm=0):
  expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

  for expert_transition in expert_dataloader:
    expert_state, expert_action = expert_transition['states'], expert_transition['actions']
    expert_action = expert_action.clamp(min=-1 + 1e-6, max=1 - 1e-6)  # Clamp expert actions to (-1, 1)

    actor_optimiser.zero_grad(set_to_none=True)
    behavioural_cloning_loss = -actor.log_prob(expert_state, expert_action).mean()  # Maximum likelihood objective
    behavioural_cloning_loss.backward()
    if max_grad_norm > 0: clip_grad_norm_(actor.parameters(), max_grad_norm)
    actor_optimiser.step()


# Performs a target estimation update
def target_estimation_update(discriminator, expert_trajectories, discriminator_optimiser, batch_size):
  expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

  for expert_transition in expert_dataloader:
    expert_state, expert_action = expert_transition['states'], expert_transition['actions']

    discriminator_optimiser.zero_grad(set_to_none=True)
    prediction, target = discriminator(expert_state, expert_action)
    regression_loss = F.mse_loss(prediction, target)
    regression_loss.backward()
    discriminator_optimiser.step()


# Performs an adversarial imitation learning update
def adversarial_imitation_update(algorithm, actor, discriminator, transitions, expert_transitions, discriminator_optimiser, r1_reg_coeff=1, pos_class_prior=1, nonnegative_margin=0):
  expert_state, expert_action, expert_next_state, expert_terminal = expert_transitions['states'], expert_transitions['actions'], expert_transitions['next_states'], expert_transitions['terminals']
  state, action, next_state, terminal = transitions['states'], transitions['actions'], transitions['next_states'], transitions['terminals']

  # TODO: Weight expert transitions even without absorbing state? https://github.com/google-research/google-research/blob/master/dac/gail.py#L109
  if algorithm in ['FAIRL', 'GAIL', 'PUGAIL']:
    D_expert = discriminator(expert_state, expert_action)
    D_policy = discriminator(state, action)
  elif algorithm == 'AIRL':
    with torch.no_grad():
      expert_log_prob = actor.log_prob(expert_state, expert_action)
      log_prob = actor.log_prob(state, action)
    D_expert = discriminator(expert_state, expert_action, expert_next_state, expert_log_prob, expert_terminal)
    D_policy = discriminator(state, action, next_state, log_prob, terminal)

  # Binary logistic regression
  discriminator_optimiser.zero_grad(set_to_none=True)
  expert_loss = (pos_class_prior if algorithm == 'PUGAIL' else 1) * F.binary_cross_entropy_with_logits(D_expert, torch.ones_like(D_expert))  # Loss on "real" (expert) data
  autograd.backward(expert_loss, create_graph=True)
  r1_reg = 0
  for param in discriminator.parameters():
    r1_reg += param.grad.norm()  # R1 gradient penalty
  if algorithm == 'PUGAIL':
    policy_loss = torch.clamp(F.binary_cross_entropy_with_logits(D_expert, torch.zeros_like(D_expert)) - pos_class_prior * F.binary_cross_entropy_with_logits(D_policy, torch.zeros_like(D_policy)), min=-nonnegative_margin)  # Loss on "real" and "unlabelled" (policy) data
  else:
    policy_loss = F.binary_cross_entropy_with_logits(D_policy, torch.zeros_like(D_policy))  # Loss on "fake" (policy) data
  (policy_loss + r1_reg_coeff * r1_reg).backward()
  discriminator_optimiser.step()
