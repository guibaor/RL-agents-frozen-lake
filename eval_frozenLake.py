import gym as gym
import numpy as np
from numpy.lib import loadtxt
import matplotlib.pyplot as plt

env = gym.make('FrozenLake8x8-v1', is_slippery = True)

total_epochs, total_penalities = 0,0

solved_episodes=[]
long_steps=[]
episodes = 100
q_table = np.loadtxt("fLake.Q.mat")


for _ in range(episodes):
    
    state = env.reset()
    epochs ,penalties, reward = 0, 0, 0
    step = 0
    done = False

    while not done:

        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward < 1 and done:
            penalties +=1
            solved_episodes.append(0)
            long_steps.append(step)
        
        epochs +=1

        if done and reward == 1:
            solved_episodes.append(1)
            long_steps.append(step)

        
        step += 1
    
    total_penalities += penalties
    total_epochs += epochs
    

print(f"Results after {episodes} episodes: ")
print(f"Number of solved episodes {str(sum(solved_episodes))}")
print(f"Number of times that agent falled on a hole: {str(solved_episodes.count(0))}")
print(f"Average timesteps per episode {total_epochs / episodes}")
print(f"Average penalties per episode {total_penalities / episodes}")
print(str(solved_episodes))
print(str(len(long_steps)))

#Plot steps over episodes
plt.style.use("bmh")
plt.plot(long_steps)
plt.title("Length of episodes over time")
plt.xlabel("Episodes")
plt.ylabel("Number of steps")
plt.show()
