import gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
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
            self.model.policy.soft_update(self.model.target_actor, self.model.actor, 1.0)
            self.model.policy.soft_update(self.model.target_critic, self.model.critic, 1.0)
        return True

def main():
    env = gym.make('CustomHopper-source-v0')

    # Create the model
    model = SAC(MlpPolicy, env, verbose=1, tensorboard_log="./sac_hopper_tensorboard/")

    # Define the hard target update frequency (e.g., every 1000 steps)
    hard_update_freq = 1000
    hard_update_callback = HardTargetUpdateCallback(update_freq=hard_update_freq)

    # Train the model with the callback
    model.learn(total_timesteps=100000, callback=hard_update_callback)

    # Save the model
    model.save("SAC_hard_target_update")

    # Optionally, load the model and evaluate
    # loaded_model = SAC.load("sac_hopper")
    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = loaded_model.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

if __name__ == "__main__":
    main()
