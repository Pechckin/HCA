from gym import make
import random
import numpy as np
from MountainCar import Continuous_MountainCarEnv
from collections import namedtuple, deque
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

Transaction = namedtuple('Transaction',
                         ('s', 'a', 'ns', 'r'))

env = Continuous_MountainCarEnv()

class Buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.appendleft(Transaction(*args))

    def sample(self, batch_size):
        batch = Transaction(*zip(*random.sample(self.memory, batch_size)))

        return Transaction(s=torch.FloatTensor(batch.s),
                           a=torch.FloatTensor(batch.a),
                           ns=torch.FloatTensor(batch.ns),
                           r=torch.FloatTensor(batch.r))

    def __len__(self):
        return len(self.memory)

    def full(self, batch_size):
        if len(self.memory) < batch_size:
            return False
        return True


# создаем скелеты двух сетей
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.tanh(x)-0.9

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256 + action_dim, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x, a):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(torch.cat([x, a], dim=1)))
        x = self.fc3(x)
        return x

class DDPG:
    def __init__(self, episodes, batch_size, mem_sz, alpha_actor,
                 alpha_critic, gamma, tau):

        states = env.observation_space.shape[0]
        actions = env.action_space.shape[0]
        self.low = env.action_space.low[0]
        self.high = env.action_space.high[0]

        self.gamma = gamma
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = Buffer(mem_sz)
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(states, actions).to(self.device).apply(self.weights)
        self.critic = Critic(states, actions).to(self.device).apply(self.weights)

        self.actor_target = Actor(states, actions).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_target = Critic(states, actions).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.alpha_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.alpha_critic)
        self.critic_criterion = nn.SmoothL1Loss()

    @staticmethod
    def weights(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    def update(self):
        if not self.memory.full(self.batch_size):
            return
        batch = self.memory.sample(self.batch_size)

        with torch.no_grad():
            na = self.actor_target(batch.ns)
        Q = self.critic(batch.s, batch.a)
        QQ = self.critic_target(batch.ns, na)
        QQQ = batch.r.unsqueeze(1) + self.gamma * QQ.detach()
        critic_loss = self.critic_criterion(Q, QQQ)
        actor_loss = -self.critic(batch.s, self.actor(batch.s)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.critic_optim.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.7)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.7)

        self.actor_optim.step()
        self.critic_optim.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def act(self, state):
        with torch.no_grad():
            action = self.actor(state).view(-1)
            #print(action)
        return action.numpy()

if __name__ == '__main__':
    agent = DDPG(episodes=10, batch_size=64, mem_sz=50000,
                 alpha_actor=1e-3, alpha_critic=1e-3, gamma=0.97,
                 tau=0.01)
    rewards = []
    best = -1000000
    for i in range(agent.episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            env.render()
            action = agent.act(torch.tensor(state, dtype=torch.float).unsqueeze(0))
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.memory.push(state, action, next_state, reward)
            state = next_state
            agent.update()


        rewards.append(total_reward)
        print(f'{i + 1})Episode reward: {total_reward} ||| Total mean {np.mean(rewards)} ||| Max score {np.max(rewards)}')
        if total_reward > best:
            best = reward
            torch.save(agent.actor, "test.pickle")






