import gym
import numpy as np
from typing import List, Tuple, TypeVar
from tminterface.interface import Client
from tminterface.client import run_client

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class TMNFEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low = -1.0, high = 1.0, shape = (2,), dtype = np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(128,128,3), dtype=np.uint8)
        self.action_freq = 50
        self.TMNFClient = TMNFClient(action_freq=self.action_freq)
        

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        command = self._action_to_command(action)
        self.TMNFClient.update_commands(command)

    def reset(self) -> ObsType:
        pass

    def render(self, mode: str = "human") -> None:
        pass

    def _action_to_command(self, action: ActType) -> List[str]:
        gas_act:str = "gas " + action[0]
        steer_act:str = "steer " + action[1]
        return [gas_act, steer_act]

class TMNFClient(Client):
    def __init__(self, action_freq = 50, verbose = True):
        super().__init__()
        self.verbose = verbose
        self.current_commands:List[str] = []
        self.action_freq = action_freq

    def on_registered(self, iface):
        if self.verbose:
            print("Client is registered on TMInterface")
    
    def on_simulation_step(self, iface, _time: int):
        if self.verbose:
            print(f"Simulation step, time = {_time}")
    
    def on_run_step(self, iface, _time: int):
        time_hint = str(_time) + "-" + str(_time+self.action_freq)+" "
        for command in self.current_commands:
            iface.execute_command(time_hint + command)
            if self.verbose:
                print(f"Executed command: {time_hint + command} at time {_time}")
    
    def update_commands(self, new_commands:List[str]):
        self.current_commands = new_commands

if __name__ == "__main__":
    env = TMNFEnv()
    print(env.action_space)
