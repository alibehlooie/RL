
import numpy as np
import matplotlib.pyplot as plt
eval_results = np.load('logs/evaluations.npz') 
rewards = eval_results['results'][:, 0]  
lengths = eval_results['results'][:, 1]
timesteps = eval_results['timesteps']   
#print top 3 in rewards array
top3 = np.sort(rewards)[::-1]
toplength = np.sort(lengths)[::-1]
toplength = toplength[:3]
top3 = top3[:3]

print(f"Rewards: {top3}")
print(f"Episode Lengths: {toplength}")
# print(f"Timesteps: {timesteps}") 



plt.figure(figsize=(10, 6))
plt.plot(timesteps, rewards)
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.title('Evaluation Rewards Over Time')
# plt.show()
plt.savefig("logs/"  + "rewardPic" + ".png")
