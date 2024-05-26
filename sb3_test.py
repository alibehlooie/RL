import argparse

import gym
from stable_baselines3 import SAC
from env.custom_hopper import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="model", type=str, help='Read model from ...')
    parser.add_argument('--n-episodes', default=100, type=int, help='Number of episodes to test')
    parser.add_argument('--render', default=True, type=bool, help='Render the environment')

    return parser.parse_args()


def main(args):
    env = gym.make('CustomHopper-source-v0')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each

    model = SAC.load(args.model)

    n_episodes = args.n_episodes
    render = args.render

    for _ in range(n_episodes):
        done = False
        state = env.reset()

        while not done:
            action, _states = model.predict(state, deterministic=True)

            state, reward, done, info = env.step(action)
            if render:
                env.render()

if __name__ == '__main__':
    args = parse_args()
    main(args)