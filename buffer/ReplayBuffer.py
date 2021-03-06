import random

class ReplayMemory(object):
    def __init__(self, type, capacity):
        self.buffer_type = type
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """save transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.buffer_type(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)