import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from agents.agent import Agent
from dataclasses import dataclass


class Policy(nn.Module):
    """
    The policy or actor network is the one that takes the state as input 
    and outputs the action to be taken.
    """
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.out = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.out(x)
        return x

class Value(nn.Module):
    """
    The value or critic network is the one that takes the state and action as input
    and outputs the value of the state-action pair.
    """
    def __init__(self, input_size, action_size, hidden_size, output_size):
       super().__init__()
       
       self.fc1 = nn.Linear(input_size + action_size, hidden_size)
       self.fc2 = nn.Linear(hidden_size, output_size)


    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

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

class DDPG_agent:
    
    def __init__(
        self, 
        observation_size,
        action_size, 
        hidden_size,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ) -> None:
        self.device = device
        
        self.actor_policy = Policy(observation_size, action_size, hidden_size)
        self.target_policy = Policy(observation_size, action_size, hidden_size)
        
        self.critic_value = Value(observation_size, action_size, hidden_size, 1)
        self.target_value = Value(observation_size, action_size, hidden_size, 1)
      
        self.noise = OUActionNoise(size=action_size)  
    
    def act(self, observation):
        observation = torch.FloatTensor(observation)
        action = self.actor_policy(observation) + torch.FloatTensor(self.noise.sample(), device=self.device)
        return action.detach().cpu().numpy()