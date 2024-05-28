"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import argparse

import numpy as np
import os
import matplotlib.pyplot as plt

import gym
from env.custom_hopper import *

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


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

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True


def main(args):
    train_env = gym.make('CustomHopper-source-v0')

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each
    
    # Create log dir
    log_dir = "./tmp/"
    os.makedirs(log_dir, exist_ok=True)

    # train_env = DummyVecEnv([lambda: train_env])
    train_env = Monitor(train_env, log_dir)
    
    model = SAC("MlpPolicy", train_env, verbose=1)

    # callback = RewardCallback(check_freq = args.callback_freq, verbose=1)
    callback = SaveOnBestTrainingRewardCallback(check_freq=args.callback_freq, log_dir=log_dir)

    timesteps = args.n_episodes

    model.learn(total_timesteps = args.n_episodes, callback=callback)
    
    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "CustomHopper")
    plt.show()

    print("Saving model to", args.save)
    model.save(args.save)

if __name__ == '__main__':
    args = parse_args()
    args.print_every = args.n_episodes/1000
    main(args)
