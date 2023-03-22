import ctypes
import time
import win32gui
import numpy as np
from typing import List
from enum import Enum
from tmai.env.utils.constants import GAME_WINDOW_NAME
import vgamepad as vg


class ArrowInput(Enum):
    UP = 0xC8
    DOWN = 0xD0
    LEFT = 0xCB
    RIGHT = 0xCD
    DEL = 0xD3

    @staticmethod
    def from_continuous_agent_out(vec: np.ndarray) -> list["ArrowInput"]:
        """
        Vector of size 2, gas and steer converted to discrete inputs
        """
        inputs = []
        if vec[0] > 0.5:
            inputs.append(ArrowInput.UP)
        elif vec[0] < -0.5:
            inputs.append(ArrowInput.DOWN)

        if vec[1] > 0.5:
            inputs.append(ArrowInput.RIGHT)

        elif vec[1] < -0.5:
            inputs.append(ArrowInput.LEFT)

        return inputs

    @staticmethod
    def from_discrete_agent_out(vec: np.ndarray) -> list["ArrowInput"]:
        "binary inpuit vector, for each action, 1 if pressed, 0 if not"
        inputs = []
        if vec[0] == 1:
            inputs.append(ArrowInput.UP)

        elif vec[1] == 1:
            inputs.append(ArrowInput.DOWN)

        if vec[2] == 1:
            inputs.append(ArrowInput.RIGHT)

        elif vec[3] == 1:
            inputs.append(ArrowInput.LEFT)

        return inputs


PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]


def refocus():
    # refocus on game window decorator, not yet used
    def wrapper(func):
        def inner(*args, **kwargs):
            hwnd = win32gui.FindWindow(None, args[0].window_name)
            win32gui.SetForegroundWindow(hwnd)
            return func(*args, **kwargs)

        return inner

    return wrapper


class KeyboardInputManager:
    def __init__(
        self, input_duration: float = 0.05, window_name: str = GAME_WINDOW_NAME
    ) -> None:
        self.input_duration = input_duration
        self.window_name = window_name

    def press_key(self, key: ArrowInput):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, key.value, 0x0008, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def release_key(self, key: ArrowInput):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, key.value, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def play_inputs(self, inputs: List[ArrowInput]):
        for input_ in inputs:
            if input_ is None:
                continue
            self.press_key(input_)
            time.sleep(self.input_duration)
            self.release_key(input_)

    def play_inputs_no_release(self, inputs: List[ArrowInput]):
        if ArrowInput.UP in inputs:
            self.press_key(ArrowInput.UP)
        else:
            self.release_key(ArrowInput.UP)

        if ArrowInput.DOWN in inputs:
            self.press_key(ArrowInput.DOWN)
        else:
            self.release_key(ArrowInput.DOWN)

        if ArrowInput.LEFT in inputs:
            self.press_key(ArrowInput.LEFT)
        else:
            self.release_key(ArrowInput.LEFT)

        if ArrowInput.RIGHT in inputs:
            self.press_key(ArrowInput.RIGHT)
        else:
            self.release_key(ArrowInput.RIGHT)


class GamepadInputManager:
    def __init__(self, window_name: str = GAME_WINDOW_NAME) -> None:
        self.gamepad = vg.VX360Gamepad()
        self.window_name = window_name

    def press_right_trigger(self, value: float):
        """
        value between 0 and 1
        """
        # print("pressing right trigger")
        self.gamepad.right_trigger_float(value_float=value)
        self.gamepad.update()

    def press_left_trigger(self, value: float):
        """
        value between 0 and 1
        """
        # print("pressing left trigger")
        self.gamepad.left_trigger_float(value_float=value)
        self.gamepad.update()

    def press_right_shoulder(self):
        """
        presses right shoulder button
        """
        # print("pressing right shoulder")
        self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        self.gamepad.update()
        time.sleep(1.0)
        self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        self.gamepad.update()

    def move_left_stick_x(self, value: float):
        self.gamepad.left_joystick_float(x_value_float=value, y_value_float=0.0)

    def play_gas(self, value: float):
        """
        value between -1 and 1
        """
        if value < 0:
            self.press_right_trigger(0.0)
            self.press_left_trigger(abs(value))
        else:
            self.press_left_trigger(0.0)
            self.press_right_trigger(value)

    def play_steer(self, value: float):
        """
        value between -1 and 1
        """
        self.move_left_stick_x(value)

    def wake_controller(self):
        self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(1.0)
        self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(1.0)
        self.gamepad.reset()


if __name__ == "__main__":
    gamepad_manager = GamepadInputManager()
    gamepad_manager.press_right_shoulder()
    print("setup done")

    while True:
        time.sleep(0.1)
        gamepad_manager.play_gas(1.0)
        time.sleep(0.1)
        gamepad_manager.play_gas(0.0)
