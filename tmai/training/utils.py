from dataclasses import dataclass
from collections import deque
import numpy as np
import random
from typing import Generic, Iterable, TypeVar

from tmai.agents.agent import Agent, RandomGamepadAgent
from tmai.env.TMNFEnv import TrackmaniaEnv

T = TypeVar("T")


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: float
    done: bool


Episode = list[Transition]


def total_reward(episode: Episode) -> float:
    return sum([t.reward for t in episode])


class Buffer(Generic[T]):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def append(self, obj: T):
        self.memory.append(obj)

    def append_multiple(self, obj_list: list[T]):
        for obj in obj_list:
            self.memory.append(obj)

    def sample(self, batch_size) -> Iterable[T]:
        return random.sample(self.memory, batch_size)

    def reset(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)


class TransitionBuffer(Buffer[Transition]):
    def __init__(self, capacity=100000):
        super().__init__(capacity)

    def append_episode(self, episode: Episode):
        self.append_multiple(episode)

    def get_batch(self, batch_size):
        batch_of_transitions = self.sample(batch_size)
        states = np.array([t.state for t in batch_of_transitions])
        actions = np.array([t.action for t in batch_of_transitions])
        next_states = np.array([t.next_state for t in batch_of_transitions])
        rewards = np.array([t.reward for t in batch_of_transitions])
        dones = np.array([t.done for t in batch_of_transitions])

        return Transition(states, actions, next_states, rewards, dones)


def play_episode(
    agent: Agent, env: TrackmaniaEnv, render=False, act_value=None
) -> Episode:
    episode = []
    observation = env.reset()
    done = False
    step = 0
    while not done:
        prev_obs = observation
        if act_value is not None:
            action = act_value()
        else:
            action = agent.act(observation)
        print(action)
        observation, reward, done, info = env.step(action)
        transition = Transition(prev_obs, action, observation, reward, done)
        episode.append(transition)
        step += 1
        if render:
            env.render()

    return episode


if __name__ == "__main__":
    env = TrackmaniaEnv(action_space="gamepad")
    agent = RandomGamepadAgent()
    while True:
        episode = play_episode(agent, env)
