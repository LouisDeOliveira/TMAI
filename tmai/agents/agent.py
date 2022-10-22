from abc import ABC
import numpy as np

from abc import ABC, abstractmethod
from typing import TypeVar

Action = TypeVar("Action")
Observation = TypeVar("Observation")


class Agent(ABC):
    @abstractmethod
    def act(self, observation: Observation) -> Action:
        raise NotImplementedError



class RandomArrowsAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space
    def act(self, observation: Observation) -> Action:
        action = self.action_space.sample()
        action[0] = 1
        action[1] = 0
        
        return action
