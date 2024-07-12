"First, let's create an enhanced ParameterizedHopperEnv that supports both SimOpt and AutoDR::"
# STEP 1 
import numpy as np
from gym.envs.mujoco import HopperEnv

class EnhancedHopperEnv(HopperEnv):
    def __init__(self, randomize_params=True):
        super().__init__()
        self.randomize_params = randomize_params
        self.original_params = {
            'mass': self.model.body_mass.copy(),
            'friction': self.model.geom_friction.copy(),
            'damping': self.model.dof_damping.copy(),
            'gravity': self.model.opt.gravity.copy()
        }
        self.current_params = self.original_params.copy()
        self.param_ranges = {
            'mass': (0.8, 1.2),
            'friction': (0.8, 1.2),
            'damping': (0.8, 1.2),
            'gravity': (0.8, 1.2)
        }

    def reset(self):
        if self.randomize_params:
            self._set_random_parameters()
        return super().reset()

    def _set_random_parameters(self):
        for key in self.current_params:
            low, high = self.param_ranges[key]
            self.current_params[key] = np.random.uniform(low, high, size=self.original_params[key].shape) * self.original_params[key]
        self._apply_parameters()

    def _apply_parameters(self):
        self.model.body_mass[:] = self.current_params['mass']
        self.model.geom_friction[:] = self.current_params['friction']
        self.model.dof_damping[:] = self.current_params['damping']
        self.model.opt.gravity[:] = self.current_params['gravity']

    def set_parameters(self, params):
        self.current_params = params
        self._apply_parameters()

    def get_parameters(self):
        return self.current_params

    def set_param_ranges(self, ranges):
        self.param_ranges = ranges

# STEP 2
"Now, let's implement a combined SimOpt and AutoDR class::"
import numpy as np
from scipy.stats import truncnorm

class SimOptAutoDR:
    def __init__(self, env, num_iterations=100, population_size=10, elite_frac=0.2, auto_dr_threshold=1000, adaptation_rate=0.1):
        self.env = env
        self.num_iterations = num_iterations
        self.population_size = population_size
        self.elite_frac = elite_frac
        self.elite_size = max(1, int(population_size * elite_frac))
        self.auto_dr_threshold = auto_dr_threshold
        self.adaptation_rate = adaptation_rate

    def optimize(self, policy, real_world_performance):
        best_params = None
        best_performance_gap = float('inf')

        for _ in range(self.num_iterations):
            population = self._generate_population()
            performances = self._evaluate_population(population, policy)
            
            performance_gaps = np.abs(performances - real_world_performance)
            
            elite_indices = np.argsort(performance_gaps)[:self.elite_size]
            elite = [population[i] for i in elite_indices]
            
            if performance_gaps[elite_indices[0]] < best_performance_gap:
                best_performance_gap = performance_gaps[elite_indices[0]]
                best_params = elite[0]
            
            self._update_param_ranges(elite, np.mean(performances))

        return best_params

    def _generate_population(self):
        population = []
        for _ in range(self.population_size):
            params = {}
            for key, (low, high) in self.env.param_ranges.items():
                params[key] = truncnorm.rvs(low, high, size=self.env.current_params[key].shape)
            population.append(params)
        return population

    def _evaluate_population(self, population, policy):
        performances = []
        for params in population:
            self.env.set_parameters(params)
            episode_reward = 0
            obs = self.env.reset()
            done = False
            while not done:
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
            performances.append(episode_reward)
        return np.array(performances)

    def _update_param_ranges(self, elite, mean_performance):
        new_ranges = {}
        for key in self.env.param_ranges:
            values = np.concatenate([params[key] for params in elite])
            mean = np.mean(values)
            std = np.std(values)
            
            if mean_performance < self.auto_dr_threshold:
                # Decrease randomization range
                new_low = max(0, mean - (1 - self.adaptation_rate) * std)
                new_high = min(2, mean + (1 - self.adaptation_rate) * std)
            else:
                # Increase randomization range
                new_low = max(0, mean - (1 + self.adaptation_rate) * std)
                new_high = min(2, mean + (1 + self.adaptation_rate) * std)
            
            new_ranges[key] = (new_low, new_high)
        
        self.env.set_param_ranges(new_ranges)

# STEP 3 
""" 
Now, let's modify the training loop to use this combined approach:
"""

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# Create the enhanced Hopper environment
env = EnhancedHopperEnv()

# Initialize the SAC model
model = SAC("MlpPolicy", env, verbose=1)

# Initialize SimOptAutoDR
sim_opt_auto_dr = SimOptAutoDR(env)

# Training loop
total_timesteps = 1000000
optimization_interval = 50000
timesteps = 0

while timesteps < total_timesteps:
    # Train for a bit
    model.learn(total_timesteps=optimization_interval, reset_num_timesteps=False)
    timesteps += optimization_interval

    # Evaluate in the current simulated environment
    mean_reward_sim, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Timesteps: {timesteps}, Sim reward: {mean_reward_sim:.2f}")

    # Simulate real-world evaluation (you would replace this with actual real-world data)
    real_world_env = EnhancedHopperEnv(randomize_params=False)
    mean_reward_real, _ = evaluate_policy(model, real_world_env, n_eval_episodes=10)
    print(f"Timesteps: {timesteps}, Real reward: {mean_reward_real:.2f}")

    # Run SimOptAutoDR
    best_params = sim_opt_auto_dr.optimize(model, mean_reward_real)
    
    # Update environment with best parameters
    env.set_parameters(best_params)

# Save the final model
model.save("sac_hopper_sim_opt_auto_dr")


#STEP 4
"EVALUATION"
# Evaluate in optimized simulation environment

env.set_parameters(best_params)
mean_reward_opt, std_reward_opt = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Optimized Sim Environment - Mean reward: {mean_reward_opt:.2f} +/- {std_reward_opt:.2f}")

# Evaluate in randomized environment
randomized_env = EnhancedHopperEnv(randomize_params=True)
mean_reward_random, std_reward_random = evaluate_policy(model, randomized_env, n_eval_episodes=100)
print(f"Randomized Environment - Mean reward: {mean_reward_random:.2f} +/- {std_reward_random:.2f}")

# Evaluate in non-randomized environment (proxy for real-world)
non_randomized_env = EnhancedHopperEnv(randomize_params=False)
mean_reward_non_random, std_reward_non_random = evaluate_policy(model, non_randomized_env, n_eval_episodes=100)
print(f"Non-randomized Environment - Mean reward: {mean_reward_non_random:.2f} +/- {std_reward_non_random:.2f}")

"""This combined approach works as follows:

SimOpt tries to find the best simulation parameters that match the real-world performance.
AutoDR adjusts the randomization ranges based on the agent's performance:

If the performance is below a threshold, it narrows the ranges to make learning easier.
If the performance is above the threshold, it widens the ranges to increase robustness.



By combining these methods, we aim to create a more adaptive and robust training process that can better bridge the sim-to-real gap. The SimOpt component helps to find a good baseline simulation, while the AutoDR component ensures that the agent is exposed to a wide range of dynamics, gradually increasing in difficulty as the agent improves.
To further analyze the results, you could:

Plot the evolution of parameter ranges over time.
Compare the performance of this combined approach with pure SimOpt, pure AutoDR, and vanilla SAC.
Analyze the final parameter distributions to understand which aspects of the environment were most crucial for matching real-world performance.
Investigate how the agent's performance changes as the randomization ranges are adjusted."""
