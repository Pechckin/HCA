from MountainCar import Continuous_MountainCarEnv
import gym
import torch
import numpy as np
env = Continuous_MountainCarEnv()

#
states = []
for i in range(10):
    done = False
    env.reset()
    while not done:
        env.render()
        action = [0.0]#[np.random.uniform(-2, 0.001)]
        next_state, reward, done, _ = env.step(action)
        states.append(next_state)

