import ctypes
from tmai.env.utils.constants import GAME_PATH, GAME_DIR, GAME_WINDOW_NAME
import subprocess
from threading import Thread
import time


class GameLauncher:
    def __init__(
        self, game_path=GAME_PATH, game_dir=GAME_DIR, game_window_name=GAME_WINDOW_NAME
    ) -> None:
        self.game_path = game_path
        self.game_dir = game_dir
        self.game_window_name = game_window_name
        self.game_thread = Thread(target=self.start_game_process, daemon=False)

    def start_game_process(self):
        subprocess.Popen(self.game_path, cwd=self.game_dir)

        while True:
            time.sleep(0)

    @property
    def game_started(self) -> bool:
        try:
            hwnd = ctypes.windll.user32.FindWindowW(None, self.game_window_name)
            if hwnd == 0:
                raise Exception("game not started")
        except:
            return False
        return True

    def start_game(self):
        if not self.game_started:
            self.game_thread.start()

        else:
            print("game already started")


if __name__ == "__main__":
    game_launcher = GameLauncher()
    game_launcher.start_game()
