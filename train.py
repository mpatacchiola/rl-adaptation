import collections
from datetime import datetime
import functools
import math
import time
from typing import Any, Callable, Dict, Optional, Sequence
import os
import random

try:
  import brax
except ImportError:
  #!pip install git+https://github.com/google/brax.git@main
  #clear_output()
  #import brax
  print("[ERROR] No Brax installation found!")
  quit()
  
from brax import envs
from brax.envs import to_torch
from brax.io import metrics
from brax.training import ppo
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from utils import Logger

import argparse
parser = argparse.ArgumentParser(description="Training script for the unsupervised phase via self-supervision")
parser.add_argument("--seed", default=-1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
parser.add_argument("--episodes_counter", default=0, type=int, help="Starts the episode counter from this value.")
parser.add_argument("--method", default="ppo", help="The name of the method")
parser.add_argument("--id", default="", help="An additional string that is attached to each saved file")
parser.add_argument("--checkpoint_path", default="", help="Path for the checkpoint file")
parser.add_argument("--env", default="HalfCheetah", help="Name of the mujoco env to use: HalfCheetah-v2, Ant-v2")
parser.add_argument("--device", default="cuda", type=str, help="The device to use (e.g. cpu, cuda, etc)")
args = parser.parse_args()

# have torch allocate on device first, to prevent JAX from swallowing up all the
# GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
#v = torch.ones(1, device='cuda')
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

if(args.seed>=0):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("[INFO] Setting SEED: " + str(args.seed))   
else:
    print("[INFO] Setting SEED: random seed")
    

class Agent(nn.Module):
  """Standard PPO Agent with GAE and observation normalization."""

  def __init__(self,
               policy_layers: Sequence[int],
               value_layers: Sequence[int],
               entropy_cost: float,
               discounting: float,
               reward_scaling: float,
               clip_ration: float,
               device: str):
    super(Agent, self).__init__()

    policy = []
    for w1, w2 in zip(policy_layers, policy_layers[1:]):
      policy.append(nn.Linear(w1, w2))
      policy.append(nn.SiLU())
    policy.pop()  # drop the final activation
    self.policy = nn.Sequential(*policy)

    value = []
    for w1, w2 in zip(value_layers, value_layers[1:]):
      value.append(nn.Linear(w1, w2))
      value.append(nn.SiLU())
    value.pop()  # drop the final activation
    self.value = nn.Sequential(*value)

    self.num_steps = torch.zeros((), device=device)
    self.running_mean = torch.zeros(policy_layers[0], device=device)
    self.running_variance = torch.zeros(policy_layers[0], device=device)

    self.entropy_cost = entropy_cost
    self.discounting = discounting
    self.reward_scaling = reward_scaling
    self.lambda_ = 0.95
    self.clip_ration = clip_ration
    self.device = device

  @torch.jit.export
  def dist_create(self, logits):
    """Normal followed by tanh.

    torch.distribution doesn't work with torch.jit, so we roll our own."""
    loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
    scale = F.softplus(scale) + .001
    return loc, scale

  @torch.jit.export
  def dist_sample_no_postprocess(self, loc, scale):
    return torch.normal(loc, scale)

  @classmethod
  def dist_postprocess(cls, x):
    return torch.tanh(x)

  @torch.jit.export
  def dist_entropy(self, loc, scale):
    log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
    entropy = 0.5 + log_normalized
    entropy = entropy * torch.ones_like(loc)
    dist = torch.normal(loc, scale)
    log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
    entropy = entropy + log_det_jacobian
    return entropy.sum(dim=-1)

  @torch.jit.export
  def dist_log_prob(self, loc, scale, dist):
    log_unnormalized = -0.5 * ((dist - loc) / scale).square()
    log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
    log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
    log_prob = log_unnormalized - log_normalized - log_det_jacobian
    return log_prob.sum(dim=-1)

  @torch.jit.export
  def update_normalization(self, observation):
    self.num_steps += observation.shape[0] * observation.shape[1]
    input_to_old_mean = observation - self.running_mean
    mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
    self.running_mean = self.running_mean + mean_diff
    input_to_new_mean = observation - self.running_mean
    var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
    self.running_variance = self.running_variance + var_diff

  @torch.jit.export
  def normalize(self, observation):
    variance = self.running_variance / (self.num_steps + 1.0)
    variance = torch.clip(variance, 1e-6, 1e6)
    return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

  @torch.jit.export
  def get_logits_action(self, observation):
    observation = self.normalize(observation)
    logits = self.policy(observation)
    loc, scale = self.dist_create(logits)
    action = self.dist_sample_no_postprocess(loc, scale)
    return logits, action

  @torch.jit.export
  def compute_gae(self, truncation, termination, reward, values,
                  bootstrap_value):
    truncation_mask = 1 - truncation
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = torch.cat(
        [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
    deltas = reward + self.discounting * (
        1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = torch.zeros_like(bootstrap_value)
    vs_minus_v_xs = torch.zeros_like(truncation_mask)

    for ti in range(truncation_mask.shape[0]):
      ti = truncation_mask.shape[0] - ti - 1
      acc = deltas[ti] + self.discounting * (
          1 - termination[ti]) * truncation_mask[ti] * self.lambda_ * acc
      vs_minus_v_xs[ti] = acc

    # Add V(x_s) to get v_s.
    vs = vs_minus_v_xs + values
    vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0)
    advantages = (reward + self.discounting *
                  (1 - termination) * vs_t_plus_1 - values) * truncation_mask
    return vs, advantages

  @torch.jit.export
  def loss(self, td: Dict[str, torch.Tensor]):
    observation = self.normalize(td['observation'])
    policy_logits = self.policy(observation[:-1])
    baseline = self.value(observation)
    baseline = torch.squeeze(baseline, dim=-1)

    # Use last baseline value (from the value function) to bootstrap.
    bootstrap_value = baseline[-1]
    baseline = baseline[:-1]
    reward = td['reward'] * self.reward_scaling
    termination = td['done'] * (1 - td['truncation'])

    loc, scale = self.dist_create(td['logits'])
    behaviour_action_log_probs = self.dist_log_prob(loc, scale, td['action'])
    loc, scale = self.dist_create(policy_logits)
    target_action_log_probs = self.dist_log_prob(loc, scale, td['action'])

    with torch.no_grad():
      vs, advantages = self.compute_gae(
          truncation=td['truncation'],
          termination=termination,
          reward=reward,
          values=baseline,
          bootstrap_value=bootstrap_value)

    rho_s = torch.exp(target_action_log_probs - behaviour_action_log_probs)
    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = rho_s.clip(1 - self.clip_ration,
                                 1 + self.clip_ration) * advantages
    policy_loss = -torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    v_error = vs - baseline
    v_loss = torch.mean(v_error * v_error) * 0.5 * 0.5

    # Entropy reward
    entropy = torch.mean(self.dist_entropy(loc, scale))
    entropy_loss = self.entropy_cost * -entropy

    return policy_loss + v_loss + entropy_loss
    

StepData = collections.namedtuple(
    'StepData',
    ('observation', 'logits', 'action', 'reward', 'done', 'truncation'))


def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
  """Map a function over each field in StepData."""
  items = {}
  keys = sds[0]._asdict().keys()
  for k in keys:
    items[k] = f(*[sd._asdict()[k] for sd in sds])
  return StepData(**items)


def eval_unroll(agent, env, length):
  """Return number of episodes and average reward for a single unroll."""
  observation = env.reset()
  episodes = torch.zeros((), device=agent.device)
  episode_reward = torch.zeros((), device=agent.device)
  for _ in range(length):
    _, action = agent.get_logits_action(observation)
    observation, reward, done, _ = env.step(Agent.dist_postprocess(action))
    episodes += torch.sum(done)
    episode_reward += torch.sum(reward)
  return episodes, episode_reward / episodes


def train_unroll(agent, env, observation, num_unrolls, unroll_length):
  """Return step data over multple unrolls."""
  sd = StepData([], [], [], [], [], [])
  for _ in range(num_unrolls):
    one_unroll = StepData([observation], [], [], [], [], [])
    for _ in range(unroll_length):
      logits, action = agent.get_logits_action(observation)
      observation, reward, done, info = env.step(Agent.dist_postprocess(action))
      one_unroll.observation.append(observation)
      one_unroll.logits.append(logits)
      one_unroll.action.append(action)
      one_unroll.reward.append(reward)
      one_unroll.done.append(done)
      one_unroll.truncation.append(info['truncation'])
    one_unroll = sd_map(torch.stack, one_unroll)
    sd = sd_map(lambda x, y: x + [y], sd, one_unroll)
  td = sd_map(torch.stack, sd)
  return observation, td


def train(
    env_name,
    num_envs: int = 2048,
    episode_length: int = 1000,
    device: str = 'cuda',
    num_timesteps: int = 30_000_000,
    eval_frequency: int = 10,
    unroll_length: int = 5,
    batch_size: int = 1024,
    num_minibatches: int = 32,
    num_update_epochs: int = 4,
    reward_scaling: float = .1,
    entropy_cost: float = 1e-2,
    discounting: float = .97,
    learning_rate: float = 3e-4,
    clip_ration: float = .3,
    seed: int = -1,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
  """Trains a policy via PPO."""
  gym_name = f'brax-{env_name}-v0'
  if gym_name not in gym.envs.registry.env_specs:
    entry_point = functools.partial(envs.create_gym_env, env_name=env_name)
    gym.register(gym_name, entry_point=entry_point)
  env = gym.make(gym_name, batch_size=num_envs, episode_length=episode_length)
  # automatically convert between jax ndarrays and torch tensors:
  env = to_torch.JaxToTorchWrapper(env, device=device)

  # env warmup
  if(seed!=-1): env.seed(seed)
  env.reset()
  action = torch.zeros(env.action_space.shape).to(device)
  env.step(action)
  
  # create the agent
  policy_layers = [
      env.observation_space.shape[-1], 64, 64, env.action_space.shape[-1] * 2
  ]
  value_layers = [env.observation_space.shape[-1], 64, 64, 1]
  agent = Agent(policy_layers=policy_layers,
                value_layers=value_layers,
                entropy_cost=entropy_cost,
                discounting=discounting,
                reward_scaling=reward_scaling,
                clip_ration=clip_ration,
                device=device)            
  agent = torch.jit.script(agent.to(device))
  optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

  sps = 0
  total_steps = 0
  total_loss = 0
  for eval_i in range(eval_frequency + 1):
    if progress_fn:
      t = time.time()
      with torch.no_grad():
        episode_count, episode_reward = eval_unroll(agent, env, episode_length)
      duration = time.time() - t
      # TODO: only count stats from completed episodes
      episode_avg_length = env.num_envs * episode_length / episode_count
      eval_sps = env.num_envs * episode_length / duration
      progress = {
          'train/num_timesteps': num_timesteps,
          'eval/episode_reward': episode_reward,
          'eval/completed_episodes': episode_count,
          'eval/avg_episode_length': episode_avg_length,
          'speed/sps': sps,
          'speed/eval_sps': eval_sps,
          'losses/total_loss': total_loss,
      }
      progress_fn(total_steps, progress)

    if eval_i == eval_frequency:
      break

    observation = env.reset()
    num_steps = batch_size * num_minibatches * unroll_length
    num_epochs = num_timesteps // (num_steps * eval_frequency)
    num_unrolls = batch_size * num_minibatches // env.num_envs
    total_loss = 0
    t = time.time()
    for _ in range(num_epochs):
      observation, td = train_unroll(agent, env, observation, num_unrolls,
                                     unroll_length)

      # make unroll first
      def unroll_first(data):
        data = data.swapaxes(0, 1)
        return data.reshape([data.shape[0], -1] + list(data.shape[3:]))
      td = sd_map(unroll_first, td)

      # update normalization statistics
      agent.update_normalization(td.observation)

      for _ in range(num_update_epochs):
        # shuffle and batch the data
        with torch.no_grad():
          permutation = torch.randperm(td.observation.shape[1], device=device)
          def shuffle_batch(data):
            data = data[:, permutation]
            data = data.reshape([data.shape[0], num_minibatches, -1] +
                                list(data.shape[2:]))
            return data.swapaxes(0, 1)
          epoch_td = sd_map(shuffle_batch, td)

        for minibatch_i in range(num_minibatches):
          td_minibatch = sd_map(lambda d: d[minibatch_i], epoch_td)
          loss = agent.loss(td_minibatch._asdict())
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          total_loss += loss

    duration = time.time() - t
    total_steps += num_epochs * num_steps
    total_loss = total_loss / (num_epochs * num_update_epochs * num_minibatches)
    sps = num_epochs * num_steps / duration

xdata = []
ydata = []
eval_sps = []
train_sps = []
times = [datetime.now()]
my_logger = Logger(header="step,reward", file_path="./logs/"+args.method.lower()+"/"+args.env.lower(), id="seed_"+str(args.seed))
    
def progress(num_steps, metrics):
  times.append(datetime.now())
  xdata.append(num_steps)
  ydata.append(metrics['eval/episode_reward'].detach().cpu().numpy())
  eval_sps.append(metrics['speed/eval_sps'])
  train_sps.append(metrics['speed/sps'])
  #clear_output(wait=True)
  plt.xlim([0, metrics['train/num_timesteps']])
  #plt.ylim([0, 6000])
  plt.xlabel('# environment steps')
  plt.ylabel('reward per episode')
  plt.plot(xdata, ydata)
  # Save the figure using same Logger name
  fig_name = my_logger.id
  plt.savefig("./logs/"+args.method.lower()+"/"+args.env.lower()+"/"+fig_name+".png", dpi=200)
  # Printing and logging
  my_logger.append(str(num_steps), str(metrics['eval/episode_reward'].item()))
  my_logger.write()
  print("[INFO]", "Step:", num_steps,
        #"Loss:", metrics['losses/total_loss'],
        "Reward:", metrics['eval/episode_reward'].item())

# Training
def main():
    print("[INFO] Environment:", args.env.lower())
    #print("[INFO] Seed:", str(SEED))

    if(args.env.lower() == 'halfcheetah'):
        train(env_name='halfcheetah',
              num_envs=2048,
              episode_length=1000,
              device=args.device,
              num_timesteps=100_000_000,
              eval_frequency=100,
              unroll_length=20,
              batch_size=512,
              num_minibatches=32,
              num_update_epochs=8,
              reward_scaling=1,
              entropy_cost=1e-2,
              discounting=.95,
              learning_rate=3e-4,
              clip_ration=.3,
              seed=args.seed,
              progress_fn=progress)
    elif(args.env.lower() == 'ant'):
        train(env_name='ant',
              num_envs=2048,
              episode_length=1000,
              device=args.device,
              num_timesteps=100_000_000,
              eval_frequency=100,
              unroll_length=5,
              batch_size=1024,
              num_minibatches=32,
              num_update_epochs=4,
              reward_scaling=10,
              entropy_cost=1e-2,
              discounting=.97,
              learning_rate=3e-4,
              clip_ration=.3,
              seed=args.seed,
              progress_fn=progress)
    else:
        print("[ERROR] The env", args.env, "is not supported")
        quit()

    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')
    print(f'eval steps/sec: {np.mean(eval_sps[1:])}')
    print(f'train steps/sec: {np.mean(train_sps[1:])}')


if __name__ == "__main__":
    main()  
