"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

import gym
from env.custom_hopper import *

from stable_baselines3 import SAC
from stable_baselines import DDPG
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=10000, type=int, help='Number of training episodes')
    # parser.add_argument('--print-every', default=10, type=int, help='Print info every <> episodes')
    parser.add_argument('--save', default="model", type=str, help='Save model as ...')

    return parser.parse_args()



def main(args):
    train_env = gym.make('CustomHopper-source-v0')

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each
    

    model = SAC("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps = args.n_episodes, log_interval = args.print_every)

    model.save(args.save)

if __name__ == '__main__':
    args = parse_args()
    args.print_every = args.n_episodes/1000
    main(args)
