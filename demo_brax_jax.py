#https://colab.research.google.com/github/google/brax/blob/main/notebooks/training.ipynb#scrollTo=25iky4mITUhh

from datetime import datetime
import functools
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

#try:
#  import brax
#except ImportError:
#  !pip install git+https://github.com/google/brax.git@main
#  clear_output()
#  import brax
import brax
  
from brax import envs
from brax import jumpy as jp
from brax.training import ppo, sac
from brax.io import html
from brax.io import model

from utils import Logger

#if 'COLAB_TPU_ADDR' in os.environ:
#  from jax.tools import colab_tpu
#  colab_tpu.setup_tpu()

#@title Preview a Brax environment { run: "auto" }
#@markdown Select the environment to train:
env_name = "ur5e"  # @param ['ant', 'humanoid', 'fetch', 'grasp', 'halfcheetah', 'ur5e', 'reacher']
env_fn = envs.create_fn(env_name=env_name)
env = env_fn()
SEED = jp.random_prngkey(seed=0)
state = env.reset(rng=SEED)
print("[INFO] Environment:", env_name)
print("[INFO] Seed:", str(SEED))
my_logger = Logger(header="step,reward", file_path="./logs/ppo/"+str(env_name))

#HTML(html.render(env.sys, [state.qp]))
  
train_fn = {
  'ant': functools.partial(
      ppo.train, num_timesteps = 30000000, log_frequency = 20,
      reward_scaling = 10, episode_length = 1000, normalize_observations = True,
      action_repeat = 1, unroll_length = 5, num_minibatches = 32,
      num_update_epochs = 4, discounting = 0.97, learning_rate = 3e-4,
      entropy_cost = 1e-2, num_envs = 2048, batch_size = 1024 # was: num_envs = 2048, batch_size = 1024
  ),
  'humanoid': functools.partial(
      ppo.train, num_timesteps = 50000000, log_frequency = 20,
      reward_scaling = 0.1, episode_length = 1000, normalize_observations = True,
      action_repeat = 1, unroll_length = 10, num_minibatches = 32,
      num_update_epochs = 8, discounting = 0.97, learning_rate = 3e-4,
      entropy_cost = 1e-3, num_envs = 2048, batch_size = 1024, seed=1
  ),
  'fetch': functools.partial(
      ppo.train, num_timesteps = 100_000_000, log_frequency = 20,
      reward_scaling = 5, episode_length = 1000, normalize_observations = True,
      action_repeat = 1, unroll_length = 20, num_minibatches = 32,
      num_update_epochs = 4, discounting = 0.997, learning_rate = 3e-4,
      entropy_cost = 0.001, num_envs = 2048, batch_size = 256
  ),
  'grasp': functools.partial(
      ppo.train, num_timesteps = 600_000_000, log_frequency = 10,
      reward_scaling = 10, episode_length = 1000, normalize_observations = True,
      action_repeat = 1, unroll_length = 20, num_minibatches = 32,
      num_update_epochs = 2, discounting = 0.99, learning_rate = 3e-4,
      entropy_cost = 0.001, num_envs = 2048, batch_size = 256
  ),
  'halfcheetah': functools.partial(
      ppo.train, num_timesteps = 100_000_000, log_frequency = 10,
      reward_scaling = 1, episode_length = 1000, normalize_observations = True,
      action_repeat = 1, unroll_length = 20, num_minibatches = 32,
      num_update_epochs = 8, discounting = 0.95, learning_rate = 3e-4,
      entropy_cost = 0.001, num_envs = 2048, batch_size = 512
  ),
  'ur5e': functools.partial(
      ppo.train, num_timesteps = 20_000_000, log_frequency = 20,
      reward_scaling = 10, episode_length = 1000, normalize_observations = True,
      action_repeat = 1, unroll_length = 5, num_minibatches = 32,
      num_update_epochs = 4, discounting = 0.95, learning_rate = 2e-4,
      entropy_cost = 1e-2, num_envs = 2048, batch_size = 1024,
      max_devices_per_host = 8
  ),
  'reacher': functools.partial(
      ppo.train, num_timesteps = 100_000_000, log_frequency = 20,
      reward_scaling = 5, episode_length = 1000, normalize_observations = True,
      action_repeat = 4, unroll_length = 50, num_minibatches = 32,
      num_update_epochs = 8, discounting = 0.95, learning_rate = 3e-4,
      entropy_cost = 1e-3, num_envs = 2048, batch_size = 256,
      max_devices_per_host = 8, seed = 1),
}[env_name]

max_y = {'ant': 6000, 
         'humanoid': 12000, 
         'fetch': 15, 
         'grasp': 100, 
         'halfcheetah': 8000,
         'ur5e': 10,
         'reacher': 5}[env_name]

min_y = {'reacher': -100}.get(env_name, 0)

# TODO: this is very important for "ant" and "halfcheetah" envs.
# It is necessary to add this line or there is an OOM break.
# This seems to be due to the pre-allocation of memory of JAX.
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# From the salina code, it seems they use other strings: 
# https://github.com/facebookresearch/salina/tree/main/salina_examples/rl/ppo_brax
#os.environ['OMP_NUM_THREADS']=1 
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']=false


xdata = []
ydata = []
times = [datetime.now()]
        
def progress(num_steps, metrics):
  times.append(datetime.now())
  xdata.append(num_steps)
  ydata.append(metrics['eval/episode_reward'])
  #clear_output(wait=True)
  plt.xlim([0, train_fn.keywords['num_timesteps']])
  plt.ylim([min_y, max_y])
  plt.xlabel('# environment steps')
  plt.ylabel('reward per episode')
  plt.plot(xdata, ydata)
  #plt.show()
  plt.savefig('./result.png', dpi=200)
  print("[INFO]", "Step:", num_steps, "Reward:", metrics['eval/episode_reward'].item())
  my_logger.append(str(num_steps), str(metrics['eval/episode_reward'].item()))
  my_logger.write()

inference_fn, params, _ = train_fn(environment_fn=env_fn, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')
