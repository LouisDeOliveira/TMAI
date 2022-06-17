from pytest import importorskip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import gym
import numpy as np
from typing import List, Tuple, TypeVar
from tminterface.interface import Client
from tminterface.client import run_client
from GameCapture import GameViewer
import cv2

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class TMNFEnv(gym.Env):
    def __init__(self, agent:"Agent"):
        self.action_space = gym.spaces.Box(low = -1.0, high = 1.0, shape = (2,), dtype = np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(128,128,3), dtype=np.uint8)
        self.action_freq = 50
        self.TMNFClient = TMNFClient(action_freq=self.action_freq, env = self, agent=agent)
        self.Viewer = GameViewer()
        self.to_render = False


    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        command = self._action_to_command(action)
        self.TMNFClient.update_commands(command)
        reward  = self.TMNFClient.reward()
        done = True if self.TMNFClient.total_reward < -30. else False
        info = {}

        return self.Viewer.get_frame(), reward, done, info


    def reset(self) -> ObsType:
        self.TMNFClient.iface.execute_command("load_state test_track.bin")
        self.TMNFClient.total_reward = 0.
        self.TMNFClient.current_commands = []
        return self.Viewer.get_frame()

        

    def render(self, mode: str = "human") -> None:
        frame =  self.Viewer.get_frame()
        cv2.imshow("Game", frame)


    def _action_to_command(self, action: ActType) -> List[str]:
        gas = -20000
        steer = np.sign(action[1])*(max(abs(action[1]*65536), 20000))
        gas_act:str = "gas " + str(gas)
        steer_act:str = "steer " + str(steer)
        return [gas_act, steer_act]

    def run_simulation(self):
        run_client(self.TMNFClient)

class TMNFClient(Client):
    def __init__(self, action_freq = 50, verbose = True, env:TMNFEnv = None, agent:"Agent" = None):
        super().__init__()
        self.verbose = verbose
        self.current_commands:List[str] = []
        self.action_freq = action_freq
        self.env = env
        self.obs = np.zeros((128,128,3))
        self.agent = agent
        self.total_reward = 0.0

    def on_registered(self, iface):
        if self.verbose:
            print("Client is registered on TMInterface")
        iface.execute_command("load_state test_track.bin")
        self.iface = iface
        self.obs = env.reset()
    
    def on_simulation_step(self, iface, _time: int):
        self.iface = iface
        if self.verbose:
            print(f"Simulation step, time = {_time}")
    
    def on_run_step(self, iface, _time: int):
        self.iface = iface
        if _time%self.action_freq == 0:
            self.obs, reward, done, _ = self.env.step(self.agent.act(self.obs))
            self.total_reward += self._reward(iface)
            print(self.total_reward)
    
           
        for cmd in self.current_commands:
            iface.execute_command(cmd)
        
        
    def update_commands(self, new_commands:List[str]):
        self.current_commands = new_commands

    def _reward(self, iface):
        speed_reward = iface.get_simulation_state().display_speed/500
        roll_reward = - abs(iface.get_simulation_state().yaw_pitch_roll[2])/3.15
        constant_reward = -0.3

        return speed_reward + roll_reward + constant_reward

    def reward(self):
        return self._reward(self.iface)

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

class Episode:
    def __init__(self) -> None:
        self.total_reward = 0.0
        self.actions = []
        self.states = []
        self.rewards = []
        self.next_states = []
if __name__ == "__main__":
    env = TMNFEnv(Agent())
    env.run_simulation()

