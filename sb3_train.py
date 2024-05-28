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
    parser.add_argument('--n-steps', default=100000, type=int, help='Number of training steps')
    # parser.add_argument('--print-every', default=10, type=int, help='Print info every <> steps')
    parser.add_argument('--save', default="models/model_test", type=str, help='Save model as ...')
    parser.add_argument('--callback-freq', default=100, type=int, help='Callback frequency')
    parser.add_argument('--verbose', default=1, type=int, help='Verbosity level for training')

    return parser.parse_args()


class RewardTrackerCallback(BaseCallback):
    """
    Custom callback for tracking rewards during training.
    """
    def __init__(self, check_freq: int = 10, verbose : int = 1):
        super(RewardTrackerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []  # List to store episode rewards
        self.verbose = verbose

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.rewards.append(self.locals["rewards"][0])  # Directly append the reward
            if self.verbose > 0:
                print(f"Episode {len(self.rewards)} - Reward: {self.rewards[-1]}")
        return True


def main(args):
    train_env = gym.make('CustomHopper-source-v0')

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each
    
   
    model = SAC("MlpPolicy", train_env, verbose = args.verbose-1)
 
    # Use the RewardTrackerCallback
    reward_tracker = RewardTrackerCallback(check_freq = args.callback_freq, verbose = args.verbose)
    eval_env = gym.make("CustomHopper-source-v0")
    eval_env = DummyVecEnv([lambda: eval_env])


    # Train the agent
    model.learn(total_timesteps=args.n_steps, callback=reward_tracker, progress_bar= True)

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_steps=10)

    # Optional: Early Stopping
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=2500, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=stop_callback, verbose=1)

    # Plot the training rewards
    plt.plot(reward_tracker.rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards Over Time")
    # plt.show()
    plt.savefig("training_rewards.png")

    # Print the evaluation results
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


    # Save the model
    print("Saving model to", args.save)
    model.save(args.save)
if __name__ == '__main__':
    args = parse_args()
    args.print_every = args.n_steps/1000
    main(args)
