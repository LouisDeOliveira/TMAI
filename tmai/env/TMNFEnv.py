from typing import TypeVar
import time
import numpy as np
from gym import Env
from gym.spaces import Box, MultiBinary

from tmai.env.TMIClient import ThreadedClient
from tmai.env.utils.GameCapture import GameViewer
from tmai.env.utils.GameInteraction import ArrowInput, KeyboardInputManager, GamepadInputManager

ArrowsActionSpace = MultiBinary((4,))  # none up down right left
ControllerActionSpace = Box(
    low=-65536, high=65536, shape=(2,), dtype=np.int32
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
        print(self.action_space.sample())
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(n_rays,), dtype=np.float32
        )
        
        self.input_manager = (KeyboardInputManager() 
                              if action_space == "arrows" 
                              else GamepadInputManager())

        self.viewer = GameViewer(n_rays=n_rays)
        self.simthread = ThreadedClient()
        self.total_reward = 0.0
        self.n_steps = 0
        self.max_steps = 1000
        self.command_frequency = 50
        self.last_action = None

    def step(self, action):
        self.last_action = action
        #plays action 
        self.action_to_command(action)
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
        self.last_action = None

        print("reset done")

        return self.viewer.get_obs()

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
        pass

    @property
    def state(self):
        return self.simthread.data

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
