import gym as gym
import numpy as np
from numpy.lib import loadtxt
import matplotlib.pyplot as plt

env = gym.make('FrozenLake8x8-v1', is_slippery = False)

total_epochs, total_penalities = 0,0
solved_episodes=[]
failed_episodes=[]
episodes = 100
q_table = np.loadtxt("fLake.Q.mat")


for _ in range(episodes):
    state = env.reset()
    epochs ,penalties, reward = 0, 0, 0
    step = 0
    done = False

    while not done:
        step += 1
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward < 0:
            penalties +=1
            failed_episodes.append(step)
        
        epochs +=1

        if done and reward == 1:
            solved_episodes.append(step)

    total_penalities += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes: ")
print(f"Number of solved episodes {str(len(solved_episodes))}")
print(f"Number of times that agent falled on a hole {str(len(failed_episodes))}")
print(f"Average timesteps per episode {total_epochs / episodes}")
print(f"Average penalties per episode {total_penalities / episodes}")

