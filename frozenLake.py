import gym as gym
import numpy as np
import random as random
import matplotlib.pyplot as plt

env = gym.make('FrozenLake8x8-v1', is_slippery = False)

print("|Actions| = ", env.action_space.n)
print("Observation space: ", env.observation_space.shape)

Q = np.zeros([env.observation_space.n,env.action_space.n])
print("Q shape: ", Q.shape)

alpha= .1  # Tasa de aprendizaje (mejor bajita)
gma = .9    # Factor de descuento de los estados futuros (entre 0 y 1)
episodes = 2000 # Numero de episodios
score_history=[]
steps = 99

for i in range (episodes) :
    
    state = env.reset()
    total_rewards = 0
    done = False
    j=0
    
    #Sarsa
    # action = np.argmax(Q[state,:]+ np.random.randn(1, env.action_space.n)*(1./(i+1)))

    while j < steps:
        
        #env.render()
        
        j+=1
        
        #Q-Learning
        action = np.argmax(Q[state,:]+ np.random.randn(1, env.action_space.n)*(1./(i+1)))
        nextState, reward, done, _ = env.step(action)
        Q[state,action] = Q[state,action] + alpha *(reward + gma * np.max(Q[nextState,:]) - Q[state,action])
        
        #Sarsa
        # nextState, reward, done, _ = env.step(action)  
        # nextAction = np.argmax(Q[nextState,:]+ np.random.randn(1, env.action_space.n)*(1./(i+1)))
        # Q[state,action] = Q[state,action] + alpha * (reward + gma * Q[nextState, nextAction] - Q[state,action])
        # action = nextAction
        

        total_rewards += reward
        
        state = nextState

        if (done== True):

            break

    score_history.append(total_rewards)

env.close()
np.savetxt("./fLake.Q.mat",Q)
    
print ("Rewards over time: " +  str(sum(score_history)/episodes))


## Plot reward over episodes
rewards_acumulated = np.cumsum(score_history)/episodes

plt.style.use("bmh")
x = np.arange(0.0, episodes, 1.0)

#plt.scatter(x, score_history)
plt.plot(x,rewards_acumulated)
plt.ylabel("Rewards")
plt.xlabel("Number of episodes")
plt.title("Rewards over episodes")
plt.show()





