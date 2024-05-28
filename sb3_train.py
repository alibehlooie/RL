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
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=5000, type=int, help='Number of training episodes')
    # parser.add_argument('--print-every', default=10, type=int, help='Print info every <> episodes')
    parser.add_argument('--save', default="models/model_test", type=str, help='Save model as ...')

    return parser.parse_args()


def main(args):
    train_env = gym.make('CustomHopper-source-v0')

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each
    
   
    model = SAC("MlpPolicy", train_env, verbose=1)

    # Train the agent (Keep this part the same)
    model.learn(total_timesteps=args.n_episodes)

    # Evaluate the trained agent
    eval_env = gym.make('CustomHopper-source-v0')
    eval_env = DummyVecEnv([lambda: eval_env])  # Wrap for SB3 evaluation
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    
    # Visualize the training rewards
    rewards = model.ep_reward_mean  # Use ep_reward_mean for rewards per episode
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards Over Time")
    plt.show()

    # Print the evaluation results
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save the model
    print("Saving model to", args.save)
    model.save(args.save)
if __name__ == '__main__':
    args = parse_args()
    args.print_every = args.n_episodes/1000
    main(args)
