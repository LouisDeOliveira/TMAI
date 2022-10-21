import ctypes
import time
import win32gui
import numpy as np
from enum import Enum
from .constants import GAME_WINDOW_NAME, GAME_WINDOW_NAME


class ArrowInputs(Enum):
    UP = 0xC8
    DOWN = 0xD0
    LEFT = 0xCB
    RIGHT = 0xCD
    DEL = 0xD3

    def from_agent_out(vec: np.ndarray) -> list["ArrowInputs"]:
        inputs = []
        if vec[0] > 0.5:
            inputs.append(ArrowInputs.UP)
        elif vec[0] < -0.5:
            inputs.append(ArrowInputs.DOWN)

        if vec[1] > 0.5:
            inputs.append(ArrowInputs.RIGHT)

        elif vec[1] < -0.5:
            inputs.append(ArrowInputs.LEFT)

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


class InputManager:
    def __init__(
        self, input_duration: float = 0.05, window_name: str = GAME_WINDOW_NAME
    ) -> None:
        self.input_duration = input_duration
        self.window_name = window_name
        self.hwnd = ctypes.windll.user32.FindWindowW(None, self.window_name)
        win32gui.SetForegroundWindow(self.hwnd)

    def press_key(self, key: ArrowInputs):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, key.value, 0x0008, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def release_key(self, key: ArrowInputs):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, key.value, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def play_inputs(self, inputs: list[ArrowInputs]):
        for input_ in inputs:
            if input_ is None:
                continue
            self.press_key(input_)
            time.sleep(self.input_duration)
            self.release_key(input_)

    def play_inputs_no_release(self, inputs: ArrowInputs):
        if ArrowInputs.UP in inputs:
            self.press_key(ArrowInputs.UP)
        else:
            self.release_key(ArrowInputs.UP)

        if ArrowInputs.DOWN in inputs:
            self.press_key(ArrowInputs.DOWN)
        else:
            self.release_key(ArrowInputs.DOWN)

        if ArrowInputs.LEFT in inputs:
            self.press_key(ArrowInputs.LEFT)
        else:
            self.release_key(ArrowInputs.LEFT)

        if ArrowInputs.RIGHT in inputs:
            self.press_key(ArrowInputs.RIGHT)
        else:
            self.release_key(ArrowInputs.RIGHT)


if __name__ == "__main__":
    input_manager = InputManager(input_duration=0.1)

    for _ in range(100):
        input_manager.play_inputs([ArrowInputs.DEL])
