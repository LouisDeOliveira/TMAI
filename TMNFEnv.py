from collections import deque
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import gym
import numpy as np
from typing import Iterable, List, Tuple, TypeVar, Generic
from tminterface.interface import Client
from tminterface.client import run_client
from GameCapture import GameViewer
import random
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.95

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
T = TypeVar("T")


class TMNFEnv(gym.Env):
    def __init__(self, agent: "Agent", value: "Value"):
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
        )
        self.action_freq = 100
        self.TMNFClient = TMNFClient(
            action_freq=self.action_freq, env=self, policy=agent, value=value
        )
        self.Viewer = GameViewer()
        self.to_render = False
        self.max_time = 30000
        self.n_steps = 0

    def step(self, action: ActType) -> Tuple[np.ndarray, float, bool, dict]:
        self.n_steps += 1
        # print(self.n_steps)
        command = self._action_to_command(action)
        self.TMNFClient.update_commands(command)
        reward = self.TMNFClient.reward()
        done = (
            True if self.n_steps > 300 or self.TMNFClient.total_reward < -10 else False
        )
        info = {}

        return self.Viewer.get_frame(), reward, done, info

    def reset(self) -> ObsType:
        self.n_steps = 0
        self.TMNFClient.total_reward = 0.0
        self.TMNFClient.current_commands = []
        self.TMNFClient.iface.execute_command("press delete")
        # if not self.TMNFClient.iface.registered:
        #     self.run_simulation()
        return self.Viewer.get_frame()

    def render(self, mode: str = "human") -> None:
        frame = self.Viewer.get_frame()
        cv2.imshow("Game", frame)

    def _action_to_command(self, action: ActType) -> List[str]:
        gas, steer = self._action_rescale(action)
        gas_act: str = "gas " + str(gas)
        steer_act: str = "steer " + str(steer)
        return [gas_act, steer_act]

    def _action_rescale(self, action: ActType) -> Tuple[float, float]:
        action += self.TMNFClient.noise.sample()
        # print(f"raw_gas = {action[0]}, raw_steer = {action[1]}")

        gas = action[0] - 1
        gas = np.clip(gas * 65536, -65536, -20000)

        abs_steer = np.abs(action[1])
        steer = np.sign(action[1]) * (min(abs_steer * 65536, 30000))

        return gas, steer

    def run_simulation(self):
        run_client(self.TMNFClient)


class TMNFClient(Client):
    def __init__(
        self,
        action_freq=100,
        verbose=True,
        env: TMNFEnv = None,
        policy: "Agent" = None,
        value: "Value" = None,
    ):
        super().__init__()
        self.verbose = verbose
        self.current_commands: List[str] = []
        self.action_freq = action_freq
        self.train_freq = 5
        self.env = env
        self.obs = np.zeros((128, 128, 3))
        self.policy = policy
        self.value = value
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=0.0001)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=0.001)
        self.total_reward = 0.0
        self.noise = OUActionNoise(size=2)
        self.transition_buffer = Buffer[Transition]()
        self.batch_size = 128
        self.episodes_counter = 0
        self.iface = None

    def on_registered(self, iface):
        if self.verbose:
            print("Client is registered on TMInterface")
        iface.execute_command("press delete")
        if self.iface is None:
            self.iface = iface
        self.iface.set_timeout(-1)
        self.iface.set_speed(1.5)
        self.obs = env.reset()

    def on_simulation_step(self, iface, _time: int):
        if self.iface is None:
            self.iface = iface
        if self.verbose:
            print(f"Simulation step, time = {_time}")

    def on_run_step(self, iface, _time: int):
        # print(f"run_step {_time}")
        if self.iface is None:
            self.iface = iface
        if _time < 300:
            return
        if _time % self.action_freq == 0:
            action = self.policy.act(self.obs)
            old_obs = np.array(self.obs.copy())
            self.obs, reward, done, _ = self.env.step(action)
            self.total_reward += self._reward(iface)

            if done:
                print(self.total_reward)
                self.episodes_counter += 1
                self.env.reset()
                if self.episodes_counter % self.train_freq == 0:
                    self.train_step()

            else:
                self.transition_buffer.append(
                    old_obs, action, np.array(self.obs.copy()), reward
                )

        for cmd in self.current_commands:
            iface.execute_command(cmd)

    def update_commands(self, new_commands: List[str]):
        self.current_commands = new_commands

    def _reward(self, iface):
        speed = iface.get_simulation_state().display_speed
        speed_reward = speed / 200 if speed > 100 else 0
        if speed < 10:
            speed_reward = -10
        roll_reward = -abs(iface.get_simulation_state().yaw_pitch_roll[2]) / 3.15
        constant_reward = -0.3

        return speed_reward + roll_reward + constant_reward

    def reward(self):
        return self._reward(self.iface)

    def train_step(
        self,
    ):
        self.value.cuda()
        self.policy.cuda()
        transitions = self.transition_buffer.sample(self.batch_size)
        states = (
            torch.cat(
                [ToTensor()(trans.state).float().unsqueeze(0) for trans in transitions],
                dim=0,
            )
            .float()
            .cuda()
        )
        actions = torch.tensor(
            np.array([trans.action for trans in transitions]),
            device=DEVICE,
            dtype=torch.float,
        )
        rewards = torch.tensor(
            np.array([trans.reward for trans in transitions]),
            device=DEVICE,
            dtype=torch.float,
        )
        next_states = (
            torch.cat(
                [
                    ToTensor()(trans.next_state).float().unsqueeze(0)
                    for trans in transitions
                ],
                dim=0,
            )
            .float()
            .cuda()
        )

        q_value = self.value(states, actions)
        next_qvalue = self.value(next_states, self.policy(next_states))
        next_qvalue = torch.squeeze(next_qvalue)
        target_qvalue = rewards + GAMMA * next_qvalue

        # print(rewards.shape)
        # print(next_qvalue.shape)

        value_loss = F.mse_loss(q_value, target_qvalue.unsqueeze(1))
        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()

        policy_loss = -self.value(states, self.policy(states)).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        print(value_loss.item(), policy_loss.item())
        self.policy.cpu()
        self.value.cpu()
        print("done training step")


class Agent(nn.Module):
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

        x = ToTensor()(x).unsqueeze(0).float()
        x = torch.Tensor(x)
        x = self.forward(x)
        x = x.squeeze(0).detach().numpy()

        return x


class Value(nn.Module):
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

    def act(self, x, y):

        x = ToTensor()(x).unsqueeze(0).float()
        y = ToTensor()(y).unsqueeze(0).float()
        z = self.forward(x, y)
        z = z.squeeze(0).detach().numpy()

        return z


class Buffer(Generic[T]):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def append(self, *args):
        transition = Transition(*args)
        self.memory.append(transition)

    def sample(self, batch_size) -> Iterable[T]:
        return random.sample(self.memory, batch_size)

    def reset(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)


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


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: float


if __name__ == "__main__":
    env = TMNFEnv(Agent(), Value())
    env.run_simulation()
