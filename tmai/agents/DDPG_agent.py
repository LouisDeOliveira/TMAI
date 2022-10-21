import torch
import torch.nn as nn
import torch.functional as F
import torchvision.transforms as Transforms
import numpy as np
from agents.agent import Agent
from dataclasses import dataclass


class Policy(nn.Module, Agent):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1, padding="same")
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 2)
        self.out = nn.Tanh()

        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.out(self.fc2(x))

        return x

    def act(self, x):

        x = Transforms.ToTensor()(x).unsqueeze(0).float()
        x = torch.Tensor(x)
        x = self.forward(x)
        x = x.squeeze(0).detach().numpy()

        return x


class Value(nn.Module, Agent):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1, padding="same")
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear(1026, 128)
        self.fc2 = nn.Linear(128, 1)

        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x, y):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 1024)
        z = torch.cat([x, y], 1)
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        return z

    def act(self, obs):
        x, y = obs
        x = Transforms.ToTensor()(x).unsqueeze(0).float()
        y = Transforms.ToTensor()(y).unsqueeze(0).float()
        z = self.forward(x, y)
        z = z.squeeze(0).detach().numpy()

        return z


class OUActionNoise:
    def __init__(
        self,
        theta=0.3,
        mu=0.0,
        sigma=0.4,
        dt=1e-2,
        x0=None,
        size=1,
        sigma_min=None,
        n_steps_annealing=1000,
    ):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.num_steps = 0

        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0
            self.c = sigma
            self.sigma_min = sigma

    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.num_steps) + self.c)
        return sigma

    def sample(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.current_sigma() * np.sqrt(self.dt) * np.random.normal(size=self.size)
        )
        self.x_prev = x
        self.num_steps += 1
        return x
