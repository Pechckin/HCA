from MountainCar import Continuous_MountainCarEnv
import gym
import torch
from sol import Actor
import numpy as np
env = Continuous_MountainCarEnv()

agent = torch.load('test.pickle')
#
print(agent(torch.tensor(env.reset(), dtype=torch.float).unsqueeze(0)).detach().numpy())
for i in range(10):
    done = False
    state = env.reset()
    while not done:
        env.render()
        action = agent(torch.tensor(state, dtype=torch.float).unsqueeze(0)).detach().numpy()

        next_state, reward, done, _ = env.step(action)


