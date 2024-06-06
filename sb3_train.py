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
from stable_baselines3.common.callbacks import CallbackList
from copy import copy, deepcopy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-steps', default=100000, type=int, help='Number of training steps')
    parser.add_argument('--save', default="models/model_test", type=str, help='Save model as ...')
    parser.add_argument('--callback-freq', default=100, type=int, help='Callback frequency')
    parser.add_argument('--verbose', default=1, type=int, help='Verbosity level for training')
    parser.add_argument('--env', default='source', type=str, help='source or target env')

    return parser.parse_args()


def RandomizedEnv(env, u = 1):
    r = np.random.uniform(low = -u, high = u, size = 3)
    env.sim.model.body_mass[1] = env.sim.model.body_mass[1] + r[0]
    env.sim.model.body_mass[2] = env.sim.model.body_mass[2] + r[1]
    env.sim.model.body_mass[3] = env.sim.model.body_mass[3] + r[2]
    return env

class DomainRandomizationCallback(BaseCallback):
    def __init__(self, u = 0.5, verbose=0):
        self.u = u
        super().__init__(verbose)   
    
    def _on_training_start(self) -> None:
        self.init_env = copy(self.training_env.envs[0].unwrapped)
        self.episode_counter = 0
        self.step_counter = 0
    
    def _on_step(self) -> bool:
        # print("On step")
        self.step_counter += 1

        # at the end of the episode
        if self.locals['dones'][0]:
            self.episode_counter += 1
            env = self.locals["env"].envs[0]
            env.model.body_mass[2] = self.init_env.model.body_mass[2] * np.random.uniform(1 - self.u, 1 + self.u)
            env.model.body_mass[3] = self.init_env.model.body_mass[3] * np.random.uniform(1 - self.u, 1 + self.u)
            env.model.body_mass[4] = self.init_env.model.body_mass[4] * np.random.uniform(1 - self.u, 1 + self.u)
            print(f"Episode {self.episode_counter} had {self.step_counter} steps -> Randomize env: ", env.model.body_mass)
            self.step_counter = 0

        return True

    def _on_training_end(self) -> None:
        print(f"Training end for {self.episode_counter} episodes")
        print(f"finally masses are: ", self.training_env.envs[0].unwrapped.model.body_mass)
        pass

randomize_callback = DomainRandomizationCallback(u=0.5)

randomize_callback = DomainRandomizationCallback()



def main(args):
    # Create a hyperparameter-grid-search
    hyperparameters = {
         "learning_rate": [1e-3],
         "gamma": [0.995],
         "tau" : [ 0.01],
         "ent_coef" : ["auto"],
         "rand_u" : [1, 0.5, 0.2]
    }
    
    for lr in hyperparameters["learning_rate"]:
        for gamma in hyperparameters["gamma"]:
            for tau in hyperparameters["tau"]:
                for ent_coef in hyperparameters["ent_coef"]:
                    for u in hyperparameters["rand_u"]:
                    
                        if(args.env == 'source'):
                            train_env = gym.make('CustomHopper-source-v0')
                            eval_env = gym.make("CustomHopper-source-v0")
                        elif(args.env == 'target'):
                            train_env = gym.make('CustomHopper-target-v0')
                            eval_env = gym.make("CustomHopper-target-v0")
                    
                        eval_env = DummyVecEnv([lambda: eval_env])
                    
                        # Randomize the environment
                        train_env = RandomizedEnv(train_env, u = u)

                        name = "SAC" + "_steps_" + str(args.n_steps) + "_lr_" + str(lr) + "_gamma_" + str(gamma) + "_tau_" + str(tau) + "_ent_coef_" + str(ent_coef) + "_u_" + str(u)
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
                        
                        randomize_callback = DomainRandomizationCallback(u = u)

                        callback = CallbackList([randomize_callback, eval_callback])

                        model = SAC("MlpPolicy", train_env, learning_rate=lr, gamma=gamma, tau=tau, ent_coef=ent_coef, verbose = args.verbose-1)
                        model.learn(total_timesteps=args.n_steps, callback=callback, progress_bar= True)
                        print("Training finished")

                        eval_results = np.load(dir_name + '/evaluations.npz') 
                        rewards = eval_results['results'][:, 0]  
                        lengths = eval_results['results'][:, 1]
                        timesteps = eval_results['timesteps']   


                        plt.figure(figsize=(10, 6))
                        plt.plot(timesteps, rewards)
                        plt.xlabel('Timesteps')
                        plt.ylabel('Mean Reward')
                        plt.title(name)
                        plt.savefig(reward_pic)


if __name__ == '__main__':
    args = parse_args()
    main(args)
