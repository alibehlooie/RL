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
        self.steps_count_list = []
        self.verbose = verbose
        self.current_episode_reward = 0
        self.steps_count = 0

    def _on_training_start(self):
        """
        This method is called before the first rollout starts.
        """
        print("Training started")
        pass

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals["rewards"][0]
        self.steps_count += 1
        
        # Check for the end of an episode
        if self.locals['dones'][0]:
            self.rewards.append(self.current_episode_reward)  # Append reward at episode end
            self.steps_count_list.append(self.steps_count) # Append steps count at episode end
            self.current_episode_reward = 0  # Reset for the next episode
            self.steps_count = 0 # Reset for the next episode
        return True
    
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # print("Rollout end (Agent finished an episode)")
        pass


def main(args):
    train_env = gym.make('CustomHopper-source-v0')

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each
    
     
    # Use the RewardTrackerCallback
    reward_tracker = RewardTrackerCallback(check_freq = args.callback_freq, verbose = args.verbose)
    eval_env = gym.make("CustomHopper-source-v0")
    eval_env = DummyVecEnv([lambda: eval_env])

    # Create a hyperparameter-grid-search
    hyperparameters = {
        # "learning_rate": [1e-3, 3e-4, 1e-4],
        "learning_rate": [ 1e-3],
        "gamma": [0.99],
        # "gamma": [0.99, 0.95, 0.9],
    }
    
    for lr in hyperparameters["learning_rate"]:
        for gamma in hyperparameters["gamma"]:
    
            print("Training with learning rate", lr, "and gamma", gamma)
    
            model = SAC("MlpPolicy", train_env, learning_rate=lr, gamma=gamma, verbose = args.verbose-1)
            model.learn(total_timesteps=args.n_steps, callback=reward_tracker, progress_bar= True)
            print("Training finished")
    
            # Plot the training rewards
            plt.plot(reward_tracker.rewards)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Training Rewards Over Time")
            name = "sac_with_rewards/training_rewards_steps" + str(args.n_steps) + "_lr_" + str(lr) + "_gamma_" + str(gamma) + ".png"
            print("Saveing plot to ", name)
            plt.savefig(name)

            # Save the model
            name = "sac_with_rewards/model_sac_steps_" + str(args.n_steps) + "lr_" + str(lr) + "_gamma_" + str(gamma) + ".zip"
            print("Saving model to ", name )
            model.save(name)
    

    # Optional: Early Stopping
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=2500, verbose=1)
    # eval_callback = EvalCallback(eval_env, callback_on_new_best=stop_callback, verbose=1)



if __name__ == '__main__':
    args = parse_args()
    main(args)
