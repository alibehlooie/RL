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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=10000, type=int, help='Number of training episodes')
    # parser.add_argument('--print-every', default=10, type=int, help='Print info every <> episodes')
    parser.add_argument('--callback-freq', default=100, type=int, help='Callback frequency')
    parser.add_argument('--save', default="model", type=str, help='Save model as ...')

    return parser.parse_args()


class RewardCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose: int = 1):
        super(RewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            reward_mean = np.mean([info['episode']['r'] for info in self.locals['infos'] if 'episode' in info])
            self.rewards.append(reward_mean)
            if self.verbose > 0:
                print(f"Step {self.n_calls}, Average Reward: {reward_mean}")
                print(self.locals['infos'])
        return True

    def _on_training_end(self) -> None:
        plt.plot(self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Average Reward Over Time')
        plt.savefig('foo.png')

def main(args):
    train_env = gym.make('CustomHopper-source-v0')

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each
    
    train_env = DummyVecEnv([lambda: train_env])

    model = SAC("MlpPolicy", train_env, verbose=1)

    callback = RewardCallback(check_freq = args.callback_freq, verbose=1) 

    model.learn(total_timesteps = args.n_episodes, callback=callback)

    print("Saving model to", args.save)
    model.save(args.save)

if __name__ == '__main__':
    args = parse_args()
    args.print_every = args.n_episodes/1000
    main(args)
