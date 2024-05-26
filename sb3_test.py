
import gym
from stable_baselines3 import SAC
from env.custom_hopper import *


env = gym.make('CustomHopper-source-v0')


def main():
    env = gym.make('CustomHopper-source-v0')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each

    model = SAC.load("models/sac_custom_hopper")
    
    

    n_episodes = 100
    render = True

    for episode in range(n_episodes):
        done = False
        state = env.reset()

        while not done:
            action, _states = model.predict(state, deterministic=True)

            state, reward, done, info = env.step(action)
            if render:
                env.render()

if __name__ == '__main__':
    main()