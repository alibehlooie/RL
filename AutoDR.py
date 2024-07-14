import numpy as np
from gym.envs.mujoco import HopperEnv
import gym
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import argparse

from copy import copy
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-steps', default=100000, type=int, help='Number of training steps')
    parser.add_argument('--callback-freq', default=100, type=int, help='Callback frequency')
    parser.add_argument('--verbose', default=1, type=int, help='Verbosity level for training')
    parser.add_argument('--env', default='source', type=str, help='source or target env')

    return parser.parse_args()

class AutoDR:
    def __init__(self, env, performance_threshold, adaptation_rate=0.1, u = 0.5):
        self.init_env = copy(env.unwrapped)
        self.env = env
        self.performance_threshold = performance_threshold
        self.adaptation_rate = adaptation_rate
        self.u = u
        self.randomization_ranges = {
            'mass': (1, 1),
            # 'friction': (1, 1)
        }

    def update_ranges(self, performance):
        if performance > self.performance_threshold:
            # Increase randomization range
            self._adjust_ranges(self.adaptation_rate)
            # only call the randomize_env function if the ranges have been updated
            print("updated ranges: ", self.randomization_ranges)
            
            print("Mass were:    ", self.env.model.body_mass[2:])
            self.randomize_env()

            print("Mass are now: ", self.env.model.body_mass[2:])
        # else remain the same            
        

    def _adjust_ranges(self, adjustment):
        for key in self.randomization_ranges:
            lower, upper = self.randomization_ranges[key]
            new_lower = min(1, max(1 - self.u, lower - adjustment))
            new_upper = max(1, min(1 + self.u, upper + adjustment))
            self.randomization_ranges[key] = (new_lower, new_upper)
        
    
    def randomize_env(self):
        # self.env.model.body_mass[2] = self.env.model.body_mass[2] * np.random.uniform(*self.randomization_ranges['mass'])
        self.env.model.body_mass[2] = self.init_env.model.body_mass[2] * np.random.uniform(*self.randomization_ranges['mass'])
        self.env.model.body_mass[3] = self.init_env.model.body_mass[3] * np.random.uniform(*self.randomization_ranges['mass'])
        self.env.model.body_mass[4] = self.init_env.model.body_mass[4] * np.random.uniform(*self.randomization_ranges['mass'])
        

    def get_randomized_env(self):
        self.env.randomize_params = True
        self.env._randomize_parameters = self._custom_randomize
        return self.env

    def _custom_randomize(self):
        mass_range = self.randomization_ranges['mass']
        friction_range = self.randomization_ranges['friction']

        mass_multipliers = np.random.uniform(*mass_range, size=self.env.model.body_mass.shape)
        self.env.model.body_mass[:] = self.env.original_masses * mass_multipliers

        friction_multipliers = np.random.uniform(*friction_range, size=self.env.model.geom_friction.shape)
        self.env.model.geom_friction[:] = self.env.original_frictions * friction_multipliers


def main(args):
    train_env = gym.make('CustomHopper-source-v0')
    eval_env = gym.make("CustomHopper-source-v0")

    # hyperparameters
    total_timesteps = args.n_steps
    eval_interval = args.callback_freq
    lr = 1e-3
    gamma = 0.995
    tau = 0.01
    ent_coef = "auto"
    u = 0.5
    threshold = 1000
    adaptation_rate = 0.005

    name = "SAC" + "_steps_" + str(args.n_steps) + "_lr_" + str(lr) + "_gamma_" + str(gamma) + "_tau_" + str(tau) + "_ent_coef_" + str(ent_coef) + "_u_" + str(u) + "_threshold_" + str(threshold) + "_callback_freq_" + str(eval_interval) + "_adaptation_rate_" + str(adaptation_rate)

    dir_name = os.path.join("AutoDR", name)
    
    # Initialize AutoDR
    # Adjust the threshold as needed
    auto_dr = AutoDR(train_env, performance_threshold = threshold, adaptation_rate = adaptation_rate)  

    if("friction" in auto_dr.randomization_ranges):
        name = name + "_friction_"
        dir_name = "AutoDR/" + name + "/"

    model = SAC("MlpPolicy", auto_dr.get_randomized_env(), learning_rate=lr, gamma=gamma, tau=tau, ent_coef=ent_coef, verbose = args.verbose-1)

    eval_results = []

    # check if model has already been trained
    if os.path.isfile(os.path.join(dir_name,  "model.zip")):
        print("Model already exists, not training again")
        return
    
    print("Training " + name)
    

    timesteps = 0 
    while timesteps < total_timesteps:
        # Train for a bit
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        timesteps += eval_interval

        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"Timesteps: {timesteps}, Mean reward: {mean_reward:.2f}")
        eval_results.append(mean_reward)

        # Update AutoDR ranges
        auto_dr.update_ranges(mean_reward)

    # Save the final model
    model.save(dir_name + "model.zip")

    # save the evaluation results
    np.save(dir_name + "eval_results.npy", eval_results)

if __name__ == "__main__":
    args = parse_args()
    main(args)