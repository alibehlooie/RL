#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 saja <saja@Saja-kubuntu>
#
# Distributed under terms of the MIT license.

"""

"""

import gymnasium as gym

from stable_baselines3 import SAC

env = gym.make("Pendulum-v1", render_mode="human")

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_pendulum")

#del model # remove to demonstrate saving and loading

#model = SAC.load("sac_pendulum")

#obs, info = env.reset()
#while True:
    #action, _states = model.predict(obs, deterministic=True)
    #obs, reward, terminated, truncated, info = env.step(action)
    ##if terminated or truncated:
        #obs, info = env.reset()
