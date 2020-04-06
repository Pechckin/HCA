from MountainCar import Continuous_MountainCarEnv
import gym
import torch
import numpy as np
from HCA import HCA

env = Continuous_MountainCarEnv()

hca = HCA(episodes=500,
          trajectory=64,
          alpha_actor=1e-3,
          alpha_credit=1e-3,
          gamma=0.99)
for i in range(hca.episodes):
    done = False
    state = env.reset()
    total_reward = 0
    while not done:
        #env.render()
        action = hca.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        hca.memory.push(state, action, next_state, reward)
        state = next_state
        hca.update()
    print(total_reward)
