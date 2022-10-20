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


class RandomAgent(Agent):
    def act(self, observation: Observation) -> Action:
        return [
            np.random.randint(low=-65000, high=-20000, size=1)[0],
            np.random.randint(low=-21000, high=21000, size=1)[0],
        ]
