"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent_actor_critic import Agent, Policy
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=50000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=1000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--save', default="model.mdl", type=str, help='Save model as ...')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

	reward_list = []

	# without tqdm to run faster
	# for episode in tqdm(range(args.n_episodes), desc='Training'):
	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
		
		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)
			reward_list.append(train_reward)

		agent.update_policy()
	
	dirname = args.save
	# make directory if it does not exist
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	model_file = os.path.join(dirname, 'model.mdl')
	# Save the model
	print(f"save the model to {model_file}")
	torch.save(agent.policy.state_dict(), model_file)
	# Save the reward list
	print(f"save the reward list to {os.path.join(dirname, 'reward_list.npy')}")
	np.save(os.path.join(dirname, 'reward_list.npy'), np.array(reward_list))

	

if __name__ == '__main__':
	main()