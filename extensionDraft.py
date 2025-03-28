"""
Implement Domain Randomization:
First, modify your Hopper environment to include randomizable parameters. These could include mass, friction, gravity, etc.
"""
import numpy as np
from gym.envs.mujoco import HopperEnv

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

        # You can add more parameter randomizations here

"""
Implement AutoDR:
AutoDR adjusts the randomization ranges based on the agent's performance. We'll create a wrapper class for this
"""

class AutoDR:
    def __init__(self, env, performance_threshold, adaptation_rate=0.1):
        self.env = env
        self.performance_threshold = performance_threshold
        self.adaptation_rate = adaptation_rate
        self.randomization_ranges = {
            'mass': (0.8, 1.2),
            'friction': (0.8, 1.2)
        }

    def update_ranges(self, performance):
        if performance < self.performance_threshold:
            # Decrease randomization range
            self._adjust_ranges(-self.adaptation_rate)
        else:
            # Increase randomization range
            self._adjust_ranges(self.adaptation_rate)

    def _adjust_ranges(self, adjustment):
        for key in self.randomization_ranges:
            lower, upper = self.randomization_ranges[key]
            new_lower = max(0, lower - adjustment)
            new_upper = min(2, upper + adjustment)
            self.randomization_ranges[key] = (new_lower, new_upper)

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

"""
step3 : Modify your training loop:
Integrate AutoDR into your SAC training process.
"""

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# Create the randomized Hopper environment
env = RandomizedHopperEnv()

# Initialize AutoDR
auto_dr = AutoDR(env, performance_threshold=1000)  # Adjust the threshold as needed

# Initialize the SAC model
model = SAC("MlpPolicy", auto_dr.get_randomized_env(), verbose=1)

# Training loop
total_timesteps = 1000000
eval_interval = 10000
timesteps = 0

while timesteps < total_timesteps:
    # Train for a bit
    model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
    timesteps += eval_interval

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Timesteps: {timesteps}, Mean reward: {mean_reward:.2f}")

    # Update AutoDR ranges
    auto_dr.update_ranges(mean_reward)

# Save the final model
model.save("sac_hopper_autodr")

"""Analysis and Evaluation:
After training, you should evaluate your model's performance in both randomized and non-randomized environments to assess the effectiveness of the AutoDR approach."""

# Evaluate in randomized environment
randomized_env = RandomizedHopperEnv(randomize_params=True)
mean_reward_random, std_reward_random = evaluate_policy(model, randomized_env, n_eval_episodes=100)
print(f"Randomized Environment - Mean reward: {mean_reward_random:.2f} +/- {std_reward_random:.2f}")

# Evaluate in non-randomized environment
non_randomized_env = RandomizedHopperEnv(randomize_params=False)
mean_reward_non_random, std_reward_non_random = evaluate_policy(model, non_randomized_env, n_eval_episodes=100)
print(f"Non-randomized Environment - Mean reward: {mean_reward_non_random:.2f} +/- {std_reward_non_random:.2f}") 
