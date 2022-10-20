from code import interact
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from typing import Generic, Union, TypeVar
from env.utils.GameCapture import GameViewer
from tminterface.client import Client
from tminterface.interface import TMInterface
import time

ArrowsActionSpace = Discrete(4, start=0)  # up down right left
ControllerActionSpace = Box(
    low=-65536, high=65536, shape=(2,), dtype=np.int32
)  # gas and steer
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class TrackmaniaEnv(Env):
    def __init__(self, action_space: "str" = "arrows", N_rays: int = 16):

        self.action_space = (
            ArrowsActionSpace if action_space == "arrows" else ControllerActionSpace
        )
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(N_rays,), dtype=np.float32
        )

        self.viewer = GameViewer(N_rays=N_rays)
        self.total_reward = 0.0
        self.n_steps = 0
        self.max_steps = 1000
        self.command_frequency = 50

    def step(self, action):
        state = self.interface.get_simulation_state()
        if state.time < 3000 and state.display_speed < 0.1:
            return self.viewer.get_obs(), 0, False, {}
        done = (
            True
            if self.n_steps >= self.max_steps or self.total_reward < -300
            else False
        )
        self.total_reward += self.reward
        self.n_steps += 1
        info = {}
        return self.viewer.get_obs(), self.reward, done, info

    def reset(self):
        print("reset")
        self.total_reward = 0.0
        self.n_steps = 0
        self._restart_race()
        self.time = 0
        print("reset done")

        return self.viewer.get_obs()

    def render(self, mode="human"):
        print(f"total reward: {self.total_reward}")
        print(f"speed: {self.speed}")
        print(f"time = {self.interface.get_simulation_state().time}")

    def action_to_command(self, action):
        if isinstance(self.action_space, Discrete):
            return self._discrete_action_to_command(action)
        elif isinstance(self.action_space, Box):
            return self._continuous_action_to_command(action)

    def _continuous_action_to_command(self, action):
        gas, steer = action
        return [f"gas {gas}", f"steer {steer}"]

    def _discrete_action_to_command(self, action):
        commands = []
        if action == 0:
            commands.append("press up")
        elif action == 1:
            commands.append("press down")
        elif action == 2:
            commands.append("press right")
        elif action == 3:
            commands.append("press left")

        return commands

    def _restart_race(self):
        self.interface.execute_command("press delete")

    @property
    def state(self):
        return self.interface.get_simulation_state()

    @property
    def speed(self):
        return self.state.display_speed

    @property
    def reward(self):
        if self.state.time < 3000:
            return 0

        speed = self.state.display_speed
        speed_reward = speed / 200 if speed > 100 else 0
        if speed < 10:
            speed_reward = -10
        roll_reward = -abs(self.state.yaw_pitch_roll[2]) / 3.15
        constant_reward = -0.3
        return speed_reward + roll_reward + constant_reward
