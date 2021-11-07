import gym as gym
import numpy as np
from numpy.lib import loadtxt
import matplotlib.pyplot as plt

env = gym.make('FrozenLake8x8-v1', is_slippery = False)

total_epochs, total_penalities = 0,0
episodes = 100
q_table = np.loadtxt("fLake.Q.mat")

for _ in range(episodes):
    state = env.reset()
    epochs ,penalties, reward = 0, 0, 0
    
    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward < 0:
            penalties +=1
        epochs +=1
    
    total_penalities += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes: ")
print(f"Average timesteps per episode {total_epochs / episodes}")
print(f"Average penalties per episode {total_penalities / episodes}")

