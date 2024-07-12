
"""Implement a parameterized Hopper environment:"""
import numpy as np
#from gym.envs.mujoco import HopperEnv
from env.custom_hopper import *
import matplotlib.pyplot as plt
import gym  
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv



class ParameterizedHopperEnv(CustomHopper):
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

    def reset(self):
        if self.randomize_params:
            self._set_random_parameters()
        return super().reset()

    def _set_random_parameters(self):
        self.current_params['mass'] = np.random.uniform(0.8, 1.2, size=self.model.body_mass.shape) * self.original_params['mass']
        self.current_params['friction'] = np.random.uniform(0.8, 1.2, size=self.model.geom_friction.shape) * self.original_params['friction']
        self.current_params['damping'] = np.random.uniform(0.8, 1.2, size=self.model.dof_damping.shape) * self.original_params['damping']
        self.current_params['gravity'] = np.random.uniform(0.8, 1.2, size=self.model.opt.gravity.shape) * self.original_params['gravity']

        self._apply_parameters()

    def _apply_parameters(self):
        self.model.body_mass[:] = self.current_params['mass']
        self.model.geom_friction[:] = self.current_params['friction']
        self.model.dof_damping[:] = self.current_params['damping']
        self.model.opt.gravity[:] = self.current_params['gravity']

    def set_parameters(self, params):
        self.current_params = params
        self._apply_parameters()


"""2.Implement SimOpt:"""
import numpy as np
from scipy.stats import truncnorm

class SimOpt:
    def __init__(self, env, num_iterations=100, population_size=10, elite_frac=0.2):
        self.env = env
        self.num_iterations = num_iterations
        self.population_size = population_size
        self.elite_frac = elite_frac
        self.elite_size = max(1, int(population_size * elite_frac))
        self.param_ranges = {
            'mass': (0.5, 1.5),
            'friction': (0.5, 1.5),
            'damping': (0.5, 1.5),
            'gravity': (0.8, 1.2)
        }

    def optimize(self, policy, real_world_performance):
        best_params = None
        best_performance_gap = float('inf')

        for _ in range(self.num_iterations):
            population = self._generate_population()
            performances = self._evaluate_population(population, policy)
            
            # Calculate performance gaps
            performance_gaps = np.abs(performances - real_world_performance)
            
            # Select elite
            elite_indices = np.argsort(performance_gaps)[:self.elite_size]
            elite = [population[i] for i in elite_indices]
            
            # Update best parameters
            if performance_gaps[elite_indices[0]] < best_performance_gap:
                best_performance_gap = performance_gaps[elite_indices[0]]
                best_params = elite[0]
            
            # Update parameter distributions
            self._update_param_ranges(elite)

        return best_params

    def _generate_population(self):
        population = []
        for _ in range(self.population_size):
            params = {}
            for key, (low, high) in self.param_ranges.items():
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

    def _update_param_ranges(self, elite):
        for key in self.param_ranges:
            values = np.concatenate([params[key] for params in elite])
            mean = np.mean(values)
            std = np.std(values)
            self.param_ranges[key] = (max(0, mean - 2*std), mean + 2*std)

"""3. Modify your training loop to incorporate SimOpt:"""
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

hyperparameters = {
         "learning_rate": [1e-3],
         "gamma": [0.995],
         "tau" : [ 0.01],
         "ent_coef" : ["auto"],
         "rand_u" : [ 0.5]
    }

# Create the parameterized Hopper environment
reward_pic = "AdoptiveDR_results/"  + "pic.png"
env = ParameterizedHopperEnv()
eval_env = gym.make("CustomHopper-source-v0")
eval_env = DummyVecEnv([lambda: eval_env])
eval_callback = EvalCallback(
    eval_env,
    n_eval_episodes=10,   # Number of episodes to evaluate 
    eval_freq=1000,       # Evaluate every 1000 steps 
    log_path= "AdoptiveDR_results/",   # Where to log results (if desired)
    best_model_save_path= "AdoptiveDR_results/", # Where to save the best model (if desired)
    # deterministic=False  # Use deterministic actions for evaluation
    )

# Initialize the SAC model
model = SAC("MlpPolicy", env, verbose=1 , 
        learning_rate=hyperparameters["learning_rate"][0], gamma=hyperparameters["gamma"][0], 
        tau=hyperparameters["tau"][0], ent_coef=hyperparameters["ent_coef"][0])

# Initialize SimOpt
sim_opt = SimOpt(env)

# Training loop
total_timesteps = 100000
sim_opt_interval = 5000
timesteps = 0

while timesteps < total_timesteps:
    # Train for a bit
    model.learn(total_timesteps=sim_opt_interval, callback = eval_callback, reset_num_timesteps=False, progress_bar= True)
 

    # Evaluate in the current simulated environment
    mean_reward_sim, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Timesteps: {timesteps}, Sim reward: {mean_reward_sim:.2f}")

    # Simulate real-world evaluation (you would replace this with actual real-world data)
    real_world_env = ParameterizedHopperEnv(randomize_params=False)
    mean_reward_real, _ = evaluate_policy(model, real_world_env, n_eval_episodes=10)
    print(f"Timesteps: {timesteps}, Real reward: {mean_reward_real:.2f}")

    # Run SimOpt
    best_params = sim_opt.optimize(model, mean_reward_real)
    
    # Update environment with best parameters
    env.set_parameters(best_params)

# Save the final model
model.save("AdaptiveDR_results/sac_hopper_simopt")
eval_results = np.load("AdoptiveDR_results/" + 'evaluations.npz') 
rewards = eval_results['results'][:, 0]  
lengths = eval_results['results'][:, 1]
timesteps = eval_results['timesteps']   


plt.figure(figsize=(10, 6))
plt.plot(timesteps, rewards)
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.title("reward_pic")
plt.savefig(reward_pic)
timesteps += sim_opt_interval

"""4.Evaluation and Analysis:After training, evaluate your model in various conditions:"""
# Evaluate in optimized simulation environment
env.set_parameters(best_params)
mean_reward_opt, std_reward_opt = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Optimized Sim Environment - Mean reward: {mean_reward_opt:.2f} +/- {std_reward_opt:.2f}")

# Evaluate in randomized environment
randomized_env = ParameterizedHopperEnv(randomize_params=True)
mean_reward_random, std_reward_random = evaluate_policy(model, randomized_env, n_eval_episodes=100)
print(f"Randomized Environment - Mean reward: {mean_reward_random:.2f} +/- {std_reward_random:.2f}")

# Evaluate in non-randomized environment (proxy for real-world)
non_randomized_env = ParameterizedHopperEnv(randomize_params=False)
mean_reward_non_random, std_reward_non_random = evaluate_policy(model, non_randomized_env, n_eval_episodes=100)
print(f"Non-randomized Environment - Mean reward: {mean_reward_non_random:.2f} +/- {std_reward_non_random:.2f}")
