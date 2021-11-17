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
    

# Training
def main():
    print("[INFO] Environment:", args.env.lower())

    # Used by the logger
    xdata = []
    ydata = []
    eval_sps = []
    train_sps = []
    times = [datetime.now()]
    my_logger = Logger(header="step,reward", file_path="./logs/"+args.method.lower()+"/"+args.env.lower(), id="seed_"+str(args.seed))
    # Logging function
    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics['eval/episode_reward'].detach().cpu().numpy())
        eval_sps.append(metrics['speed/eval_sps'])
        train_sps.append(metrics['speed/sps'])
        plt.xlim([0, metrics['train/num_timesteps']])
        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.plot(xdata, ydata)
        # Save the figure using same Logger name
        fig_name = my_logger.id
        plt.savefig("./logs/"+args.method.lower()+"/"+args.env.lower()+"/"+fig_name+".png", dpi=200)
        # Printing and logging
        my_logger.append(str(num_steps), str(metrics['eval/episode_reward'].item()))
        my_logger.write()
        print("[INFO]", "Step:", num_steps, "Reward:", metrics['eval/episode_reward'].item())
        
    # Load the model and training parameters
    if(args.method.lower()=="ppo"):
        from methods.ppo import PPO
        model = PPO()
        if(args.env.lower() == 'ant'):
            params = {"env_name": "ant", "num_envs": 2048, "episode_length": 1000, "device": args.device, 
              "num_timesteps": 100_000_000, "eval_frequency": 100, "unroll_length": 5, "batch_size": 1024, 
              "num_minibatches": 32, "num_update_epochs": 4, "reward_scaling": 10, "entropy_cost": 1e-2, 
              "discounting": .97, "learning_rate": 3e-4, "clip_ration": .3, "seed": args.seed, "progress_fn": progress}
        elif(args.env.lower() == 'halfcheetah'):
            params = {"env_name": "halfcheetah", "num_envs": 2048, "episode_length": 1000, "device": args.device, 
              "num_timesteps": 100_000_000, "eval_frequency": 100, "unroll_length": 20, "batch_size": 512, 
              "num_minibatches": 32, "num_update_epochs": 8, "reward_scaling": 1, "entropy_cost": 1e-2, 
              "discounting": .95, "learning_rate": 3e-4, "clip_ration": .3, "seed": args.seed, "progress_fn": progress}
        else:
            print("[ERROR] The method", args.method, "does not support the env", args.env)
            quit()
    else:
        print("[ERROR] The method", args.method, "is not supported!")


    # Start the training
    model.train(**params)
    
    # Saving
    checkpoint_path = "./checkpoints/" + args.method.lower() + "/" + args.env.lower()
    if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)
    model.save(checkpoint_path + "/" + my_logger.id + ".dat")

    # Printing stuff
    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')
    print(f'eval steps/sec: {np.mean(eval_sps[1:])}')
    print(f'train steps/sec: {np.mean(train_sps[1:])}')


if __name__ == "__main__":
    main()  
