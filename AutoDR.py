import numpy as np
from gym.envs.mujoco import HopperEnv
import gym
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-steps', default=100000, type=int, help='Number of training steps')
    parser.add_argument('--callback-freq', default=100, type=int, help='Callback frequency')
    parser.add_argument('--verbose', default=1, type=int, help='Verbosity level for training')
    parser.add_argument('--env', default='source', type=str, help='source or target env')

    return parser.parse_args()


class RandomizedHopperEnv(HopperEnv):
    def __init__(self, randomize_params=True):
        super().__init__()
        self.randomize_params = randomize_params
        self.original_masses = self.model.body_mass.copy()
        self.original_frictions = self.model.geom_friction.copy()

    def reset(self):
        if self.randomize_params:
            self._randomize_parameters()
        return super().reset()

    def _randomize_parameters(self):
        # Randomize masses
        mass_multipliers = np.random.uniform(0.8, 1.2, size=self.model.body_mass.shape)
        self.model.body_mass[:] = self.original_masses * mass_multipliers

        # Randomize friction
        friction_multipliers = np.random.uniform(0.8, 1.2, size=self.model.geom_friction.shape)
        self.model.geom_friction[:] = self.original_frictions * friction_multipliers


class AutoDR:
    def __init__(self, env, performance_threshold, adaptation_rate=0.1):
        self.init_env = env
        self.env = env
        self.performance_threshold = performance_threshold
        self.adaptation_rate = adaptation_rate
        self.randomization_ranges = {
            'mass': (0.8, 1.2),
            'friction': (0.8, 1.2)
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
        # else:
            # Decrease randomization range
            # if we are lower than the threshold, keep the weights as they are -> set the range to (1, 1)
            # self.randomization_ranges = {
                # 'mass': (1, 1),
                # 'friction': (1, 1)
            # }
            
        
        

    def _adjust_ranges(self, adjustment):
        for key in self.randomization_ranges:
            lower, upper = self.randomization_ranges[key]
            new_lower = min(1, max(0, lower - adjustment))
            new_upper = max(1, min(2, upper + adjustment))
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
    
    # env = gym.make('CustomHopper-source-v0')
    # eval_env = gym.make("CustomHopper-source-v0")

    total_timesteps = 10000
    eval_interval = 100
    timesteps = 0

    if(args.env == 'source'):
        train_env = gym.make('CustomHopper-source-v0')
        eval_env = gym.make("CustomHopper-source-v0")
    elif(args.env == 'target'):
        train_env = gym.make('CustomHopper-target-v0')
        eval_env = gym.make("CustomHopper-target-v0")

    lr = 1e-3
    gamma = 0.995
    tau = 0.01
    ent_coef = "auto"

    name = "SAC" + "_steps_" + str(args.n_steps) + "_lr_" + str(lr) + "_gamma_" + str(gamma) + "_tau_" + str(tau) + "_ent_coef_" + str(ent_coef) + "AutoDR"

    dir_name = "AutoDR/" + name + "/"
    
    # Initialize AutoDR
    auto_dr = AutoDR(train_env, performance_threshold = 800, adaptation_rate=0.05)  # Adjust the threshold as needed


    model = SAC("MlpPolicy", auto_dr.get_randomized_env(), verbose=1)

    # eval_callback = EvalCallback(
    #     eval_env,
    #     n_eval_episodes=10,   # Number of episodes to evaluate 
    #     eval_freq=1000,       # Evaluate every 1000 steps 
    #     log_path= dir_name,   # Where to log results (if desired)
    #     best_model_save_path= dir_name, # Where to save the best model (if desired)
    #     deterministic=True    # Use deterministic actions for evaluation
    #     )

    # model = SAC("MlpPolicy", train_env, learning_rate=lr, gamma=gamma, tau=tau, ent_coef=ent_coef, verbose = args.verbose-1)

    eval_results = []

    while timesteps < total_timesteps:
        # Train for a bit
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        timesteps += eval_interval

        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
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