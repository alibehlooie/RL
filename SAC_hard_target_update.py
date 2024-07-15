import gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import gym
from env.custom_hopper import *
from stable_baselines3 import SAC


class HardTargetUpdateCallback(BaseCallback):
    def __init__(self, update_freq, verbose=0):
        super(HardTargetUpdateCallback, self).__init__(verbose)
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        # Perform hard target update every `update_freq` steps
        if self.n_calls % self.update_freq == 0:
            self.model.critic_target.load_state_dict(self.model.critic.state_dict())
            self.model.actor.load_state_dict(self.model.actor.state_dict())
            if self.verbose > 1:
                print(f"Performing hard update at step {self.n_calls}")
        return True

def main():
    # Create the environment
    env = gym.make('CustomHopper-source-v0')
    eval_env = gym.make('CustomHopper-source-v0')

    dir_name = "SAC_HARD/best_params/"
    # Create the model
    model = SAC(MlpPolicy, env, verbose=1, learning_rate = 1e-3, gamma=0.995, tau=0.01, ent_coef="auto", tensorboard_log="./sac_hopper_tensorboard/")

    # Define the hard target update frequency (e.g., every 1000 steps)
    hard_update_freq = 1000
    hard_update_callback = HardTargetUpdateCallback(update_freq=hard_update_freq)

    eval_callback = EvalCallback(
                            eval_env,
                            n_eval_episodes = 10,   # Number of episodes to evaluate 
                            eval_freq = 1000,       # Evaluate every 1000 steps 
                            log_path = dir_name,   # Where to log results (if desired)
                            best_model_save_path = dir_name, # Where to save the best model (if desired)
                            deterministic = True    # Use deterministic actions for evaluation
                            )
    

    callback = CallbackList([eval_callback, hard_update_callback])
    # Train the model with the callback
    model.learn(total_timesteps=100000, callback=callback, progress_bar=True)

    # Save the model
    model.save("SAC_HARD/best_params/model")


if __name__ == "__main__":
    main()