import torch
import random
import numpy as np
import torch.nn.functional as F

from model import LinearQNet
from replay_memory import ReplayMemory


class Agent:
    def __init__(self, input_size, hidden_size, output_size, lr, gamma):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = gamma
        self.memory = ReplayMemory(100_000)
        self.model = LinearQNet(input_size, hidden_size, output_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            return random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            return [int(i == torch.argmax(prediction)) for i in range(3)]

    def remember(self, *experience):
        self.memory.push(experience)

    def train_long_memory(self, batch_size=100):
        if len(self.memory) < batch_size:
            return
        mini_batch = self.memory.sample(batch_size)
        self._train_batch(mini_batch)

    def _train_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)

        q_pred = self.model(states).gather(1, torch.argmax(actions, dim=1).unsqueeze(1)).squeeze()
        q_target = rewards + self.gamma * self.model(next_states).max(1)[0] * (1 - torch.tensor(dones, dtype=torch.float))

        loss = F.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
