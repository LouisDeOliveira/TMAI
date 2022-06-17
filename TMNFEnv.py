from msilib.schema import SelfReg
from pdb import runcall
from tkinter.tix import Tree
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
    def __init__(self):
        self.action_space = gym.spaces.Box(low = -1.0, high = 1.0, shape = (2,), dtype = np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(128,128,3), dtype=np.uint8)
        self.action_freq = 50
        self.TMNFClient = TMNFClient(action_freq=self.action_freq, env = self)
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
        self.TMNFClient = TMNFClient(action_freq=self.action_freq, env = self)

        

    def render(self, mode: str = "human") -> None:
        frame =  self.Viewer.get_frame()
        cv2.imshow("Game", frame)


    def _action_to_command(self, action: ActType) -> List[str]:
        gas_act:str = "gas " + str(action[0]*65536)
        steer_act:str = "steer " + str(action[1]*65536)
        return [gas_act, steer_act]

    def run_simulation(self):
        run_client(self.TMNFClient)

class TMNFClient(Client):
    def __init__(self, action_freq = 50, verbose = True, env = None):
        super().__init__()
        self.verbose = verbose
        self.current_commands:List[str] = []
        self.action_freq = action_freq
        self.env = env
        self.total_reward = 0.0

    def on_registered(self, iface):
        if self.verbose:
            print("Client is registered on TMInterface")
        iface.execute_command("load_state test_track.bin")
        self.iface = iface
    
    def on_simulation_step(self, iface, _time: int):
        if self.verbose:
            print(f"Simulation step, time = {_time}")
    
    def on_run_step(self, iface, _time: int):
        if _time%self.action_freq == 0:
            self.env.step(np.array([-1.,0.]))
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




if __name__ == "__main__":
    env = TMNFEnv()
    env.run_simulation()

