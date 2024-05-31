"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
import os

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

    # print('State space:', train_env.observation_space)  # state-space
    # print('Action space:', train_env.action_space)  # action-space
    # print('Dynamics parameters:', train_env.get_parameters())  # masses of each
    
     
    # Use the RewardTrackerCallback
    reward_tracker = RewardTrackerCallback(check_freq = args.callback_freq, verbose = args.verbose)

    # Create a hyperparameter-grid-search
    hyperparameters = {
         "learning_rate": [1e-3],
         "gamma": [0.99],
         "tau" : [0.005],
         "ent_coef" : ["auto"],
    }
    
    for lr in hyperparameters["learning_rate"]:
        for gamma in hyperparameters["gamma"]:
            for tau in hyperparameters["tau"]:
                for ent_coef in hyperparameters["ent_coef"]:
                    
                    train_env = gym.make('CustomHopper-source-v0')

                    eval_env = gym.make("CustomHopper-source-v0")
                    eval_env = DummyVecEnv([lambda: eval_env])
                    
                    name = "SAC" + "_steps_" + str(args.n_steps) + "_lr_" + str(lr) + "_gamma_" + str(gamma) + "_tau_" + str(tau) + "_ent_coef_" + str(ent_coef)
                    print("Training " + name)
                    
                    dir_name = "SAC-hyper-eval_callback/" + name + "/"
                    reward_pic = "Reward_Pics/" +  name + ".png"
                    
                    # check if rewardPic file already exists
                    if os.path.isfile(reward_pic):
                        print("File already exists, not training again")
                        continue

                    eval_callback = EvalCallback(
                        eval_env,
                        n_eval_episodes=10,   # Number of episodes to evaluate 
                        eval_freq=1000,       # Evaluate every 1000 steps 
                        log_path= dir_name,   # Where to log results (if desired)
                        best_model_save_path= dir_name, # Where to save the best model (if desired)
                        deterministic=True    # Use deterministic actions for evaluation
                        )

    
                    model = SAC("MlpPolicy", train_env, learning_rate=lr, gamma=gamma, tau=tau, ent_coef=ent_coef, verbose = args.verbose-1)
                    model.learn(total_timesteps=args.n_steps, callback=eval_callback, progress_bar= True)
                    print("Training finished")

                    eval_results = np.load(dir_name + '/evaluations.npz') 
                    rewards = eval_results['results'][:, 0]  
                    lengths = eval_results['results'][:, 1]
                    timesteps = eval_results['timesteps']   


                    plt.figure(figsize=(10, 6))
                    plt.plot(timesteps, rewards)
                    plt.xlabel('Timesteps')
                    plt.ylabel('Mean Reward')
                    plt.title('Evaluation Rewards Over Time')
                    plt.savefig(reward_pic)


    # Optional: Early Stopping
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=2500, verbose=1)
    # eval_callback = EvalCallback(eval_env, callback_on_new_best=stop_callback, verbose=1)



if __name__ == '__main__':
    args = parse_args()
    main(args)
