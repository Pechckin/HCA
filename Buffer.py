import random
from collections import namedtuple, deque
import torch


class Buffer(object):
    def __init__(self, capacity):
        self.Transaction = namedtuple('Transaction',
                                      ('s', 'a', 'ns', 'r'))
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(self.Transaction(*args))

    def sample(self):
        batch = self.Transaction(*zip(*self.memory))
        self.memory.clear()
        return self.Transaction(s=torch.FloatTensor(batch.s),
                                a=torch.FloatTensor(batch.a),
                                ns=torch.FloatTensor(batch.ns),
                                r=torch.FloatTensor(batch.r))

    def full(self):
        if len(self.memory) < self.capacity:
            return False
        return True
