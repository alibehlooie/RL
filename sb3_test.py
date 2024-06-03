import argparse

import gym

from stable_baselines3 import SAC
from env.custom_hopper import *
from gym import Wrapper
from mujoco_py import GlfwContext
import glfw
import matplotlib.pyplot as plt
import numpy as np


# class CameraControlWrapper(Wrapper):
#     def __init__(self, env):
#         super().__init__(env)

#     def render(self, mode="human", **kwargs):
#         # Modify the camera angle here before calling the original render method
#         # For example:
#         # self.unwrapped.viewer.cam.elevation = -45  # Set the camera elevation angle

#            return self.env.render(mode, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="model", type=str, help='Read model from ...')
    parser.add_argument('--n-episodes', default=100, type=int, help='Number of episodes to test')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--no-render', dest='render', action='store_false', help='Do not render the simulator')
    parser.add_argument('--env', default='source', type=str, help='source or target env')

    return parser.parse_args()

def main(args):

    if(args.env == 'source'):
        env = gym.make('CustomHopper-source-v0')
    elif(args.env == 'target'):
        env = gym.make('CustomHopper-target-v0')
    else :
        print('Invalid environment')
        return

    
    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each

    model = SAC.load(args.model)

    n_episodes = args.n_episodes
    render = args.render
    rewards_list = np.array([])

    for i in range(n_episodes):
        done = False
        state = env.reset()
        total_reward = 0

        while not done:
            action, _states = model.predict(state, deterministic=True)

            state, reward, done, info = env.step(action)
            total_reward += reward
            
            if render:
                env.render("human")
                glfw.poll_events()

        print(f"Episode {i} reward: {total_reward}")
        rewards_list = np.append(rewards_list, total_reward)

    plt.hist(rewards_list, bins=20)
    plt.show()
    plt.savefig('rewards.png')
        

if __name__ == '__main__':
    args = parse_args()
    main(args)