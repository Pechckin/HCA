from Buffer import Buffer
from Nets import Policy, Credit
import torch
from MountainCar import Continuous_MountainCarEnv
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

env = Continuous_MountainCarEnv()


class HCA:
    def __init__(self, episodes, trajectory, alpha_actor, alpha_credit, gamma):

        states = env.observation_space.shape[0]
        actions = env.action_space.shape[0]
        self.low = env.action_space.low[0]
        self.high = env.action_space.high[0]

        self.gamma = gamma
        self.alpha_actor = alpha_actor
        self.alpha_credit = alpha_credit
        self.episodes = episodes
        self.memory = Buffer(trajectory)

        self.policy = Policy(states, actions).apply(self.weights)
        self.credit = Credit(states, actions).apply(self.weights)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.alpha_actor)
        self.credit_optim = optim.Adam(self.credit.parameters(), lr=self.alpha_credit)
        self.credit_loss = nn.CrossEntropyLoss()

    def discount(self, rewards):
        R = 0.0
        returns = []
        for r in rewards.numpy()[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-10)
        return returns

    @staticmethod
    def weights(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    def update(self):
        if not self.memory.full():
            return
        batch = self.memory.sample()

        Zs = self.discount(batch.r)

        # Policy
        mu_policy, sigma_policy = self.policy(batch.s)
        log_prob_policy = Normal(mu_policy, sigma_policy).log_prob(batch.a).mean(dim=1, keepdims=True)

        # Credit
        mu_credit, sigma_credit = self.credit(batch.s, Zs)
        log_prob_credit = Normal(mu_credit, sigma_credit).log_prob(batch.a).mean(dim=1, keepdims=True)


        ratio = torch.exp(log_prob_policy - log_prob_credit.detach())
        A = (1 - ratio) * Zs.unsqueeze(1)
        policy_loss = -(A.T @ log_prob_policy) / batch.r.size(0)
        self.policy_optim.zero_grad()
        policy_loss.backward()

        credit_loss = -torch.mean(log_prob_policy.detach() * log_prob_credit)

        self.credit_optim.zero_grad()
        credit_loss.backward()


        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.7)
        torch.nn.utils.clip_grad_norm_(self.credit.parameters(), 0.7)

        self.policy_optim.step()
        self.credit_optim.step()

    def act(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            mu, sigma = self.policy(state)
        return torch.clamp(Normal(mu, sigma).sample(), min=self.low, max=self.high)
