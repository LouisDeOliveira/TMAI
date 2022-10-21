from dataclasses import dataclass
from collections import deque
import numpy as np
import random
from typing import Generic, Iterable, TypeVar

from tmai.agents.agent import Agent, RandomArrowsAgent
from tmai.env.TMIClient import ThreadedClient
from tmai.env.TMNFEnv import TrackmaniaEnv

T = TypeVar("T")

@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: float

Episode = list[Transition]

class Buffer(Generic[T]):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def append(self, *args):
        transition = T(*args)
        self.memory.append(transition)

    def sample(self, batch_size) -> Iterable[T]:
        return random.sample(self.memory, batch_size)

    def reset(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)


def play_episode(agent:Agent, env:TrackmaniaEnv) -> Episode:
    env = TrackmaniaEnv(action_space="arrows")
    episode = []
    observation = env.reset()
    done = False
    step = 0
    while not done:
        prev_obs = observation
        action = agent.act(observation)
        print(action)
        observation, reward, done, info = env.step(action)
        transition = Transition(prev_obs, action, observation, reward)
        episode.append(transition)
        step += 1
        env.render()

    return episode

if __name__ == "__main__":
    episode = play_episode(RandomArrowsAgent(), TrackmaniaEnv(simthread=ThreadedClient(), action_space="arrows"))
    print(len(episode))
