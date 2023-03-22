import time
from typing import TypeVar

import numpy as np
from gym import Env
from gym.spaces import Box, MultiBinary

from tmai.env.TMIClient import ThreadedClient
from tmai.env.utils.GameCapture import GameViewer
from tmai.env.utils.GameInteraction import (
    ArrowInput,
    GamepadInputManager,
    KeyboardInputManager,
)
from tmai.env.utils.GameLaunch import GameLauncher

ArrowsActionSpace = MultiBinary((4,))  # none up down right left
ControllerActionSpace = Box(
    low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32
)  # gas and steer
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class TrackmaniaEnv(Env):
    """
    Gym env interfacing the game.
    Observations are the rays of the game viewer.
    Controls are the arrow keys or the gas and steer.
    """

    def __init__(
        self,
        action_space: str = "arrows",
        n_rays: int = 16,
    ):
        self.action_space = (
            ArrowsActionSpace if action_space == "arrows" else ControllerActionSpace
        )
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(n_rays + 1,), dtype=np.float32
        )

        self.input_manager = (
            KeyboardInputManager()
            if action_space == "arrows"
            else GamepadInputManager()
        )

        game_launcher = GameLauncher()
        if not game_launcher.game_started:
            game_launcher.start_game()
            print("game started")
            input("press enter when game is ready")
            time.sleep(4)

        self.viewer = GameViewer(n_rays=n_rays)
        self.simthread = ThreadedClient()
        self.total_reward = 0.0
        self.n_steps = 0
        self.max_steps = 1000
        self.command_frequency = 50
        self.last_action = None
        self.low_speed_steps = 0

    def step(self, action):
        self.last_action = action
        # plays action
        self.action_to_command(action)
        done = (
            True
            if self.n_steps >= self.max_steps or self.total_reward < -300
            else False
        )
        self.total_reward += self.reward
        self.n_steps += 1
        info = {}
        time.sleep(self.command_frequency * 10e-3)
        return self.observation, self.reward, done, info

    def reset(self):
        print("reset")
        self.total_reward = 0.0
        self.n_steps = 0
        self._restart_race()
        self.time = 0
        self.last_action = None
        self.low_speed_steps = 0
        print("reset done")

        return self.observation

    def render(self, mode="human"):
        print(f"total reward: {self.total_reward}")
        print(f"speed: {self.speed}")
        print(f"time = {self.state.time}")

    def action_to_command(self, action):
        if isinstance(self.action_space, MultiBinary):
            return self._discrete_action_to_command(action)
        elif isinstance(self.action_space, Box):
            return self._continuous_action_to_command(action)

    def _continuous_action_to_command(self, action):
        gas, steer = action
        self.input_manager.play_gas(gas)
        self.input_manager.play_steer(steer)

    def _discrete_action_to_command(self, action):
        commands = ArrowInput.from_discrete_agent_out(action)
        self.input_manager.play_inputs_no_release(commands)

    def _restart_race(self):
        if isinstance(self.input_manager, KeyboardInputManager):
            self._keyboard_restart()
        else:
            self._gamepad_restart()

    def _keyboard_restart(self):
        self.input_manager.press_key(ArrowInput.DEL)
        time.sleep(0.1)
        self.input_manager.release_key(ArrowInput.DEL)

    def _gamepad_restart(self):
        self.input_manager.press_right_shoulder()

    @property
    def state(self):
        return self.simthread.data

    @property
    def speed(self):
        return self.state.display_speed

    @property
    def observation(self):
        return np.concatenate([self.viewer.get_obs(), [self.speed / 400]])

    @property
    def reward(self):
        speed = self.speed
        if self.state.time < 3000:
            return 0

        speed_reward = speed / 400
        roll_reward = -abs(self.state.yaw_pitch_roll[2]) / 3.15
        constant_reward = -0.3
        gas_reward = self.last_action[0] * 2

        if self.last_action[0] < 0:
            constant_reward -= 10
            gas_reward = 0

        if min(self.observation) < 0.06:
            constant_reward -= 100

        elif 10 < speed < 100:
            speed_reward = -1
            gas_reward = 0

        elif speed < 10:
            self.low_speed_steps += 1
            speed_reward = -5 * self.low_speed_steps
            gas_reward = 0

        else:
            self.low_speed_steps = 0

        return speed_reward + roll_reward + constant_reward + gas_reward
