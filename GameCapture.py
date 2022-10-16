import numpy as np
import cv2
from mss import mss
import os
import win32.win32gui as wind32


def getWindowGeometry(name: str) -> tuple:
    """
    Get the geometry of a window.
    """
    hwnd = wind32.FindWindow(None, name)
    left, top, right, bottom = wind32.GetWindowRect(hwnd)

    return left + 10, top + 40, right - 10, bottom - 10


class GameViewer:
    def __init__(self) -> None:

        self.window_name = "TrackMania Nations Forever (TMInterface 1.1.1)"
        self.sct = mss()

    @property
    def bounding_box(self):
        return getWindowGeometry(self.window_name)

    def process_screen(self, screenshot: np.ndarray, show_rays=False) -> np.ndarray:
        baw = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        baw = cv2.Canny(baw, threshold1=100, threshold2=300)
        element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
        baw = cv2.dilate(baw, element, iterations=3)
        baw = cv2.GaussianBlur(baw, (3, 3), 0)

        baw = cv2.resize(baw, (128, 128))
        height = len(baw)
        cut = baw[height // 2 : height // 2 + 32, :]
        return cut

    def is_inbouds(self, x, y, frame):
        return x >= 0 and x < len(frame[0]) and y >= 0 and y < len(frame)

    def find_end(self, direction, frame):
        """
        Finds the pixel of contact between the ray and the border of the image
        starting from the bottom center of the frame
        """
        dx = np.cos(direction)
        dy = np.sin(direction)

        cur_x = len(frame[0]) // 2
        cur_y = len(frame) - 1

        while (
            self.is_inbouds(cur_x, cur_y, frame) and frame[int(cur_y)][int(cur_x)] == 0
        ):
            cur_x += dx
            cur_y -= dy

        return [int(cur_x), int(cur_y)]

    def get_distance(self, processed_img, direction, ref_size):
        collision = self.find_end(direction, processed_img)
        return np.hypot(*collision) / ref_size

    def get_obs(self, N_rays=15):
        processed_img = self.get_frame()
        ref_size = np.hypot(processed_img.shape[0], processed_img.shape[1])
        collisions = np.zeros(N_rays)
        for i in range(N_rays):
            direction = np.pi * i / N_rays
            collisions[i] = self.get_distance(processed_img, direction, ref_size)

        return collisions.astype(np.float32)

    def get_frame(
        self,
        size=(256, 256),
    ) -> np.ndarray:
        """
        Pulls a frame from the game and processes it

        Args:
            size (tuple, optional): size to resize the screenshot to. Defaults to (256, 256).

        Returns:
            np.ndarray: processed frame
        """
        sct_img = cv2.resize(
            self.get_raw_frame(),
            size,
        )
        sct_img = self.process_screen(sct_img)

        return sct_img

    def get_raw_frame(self):
        """
        Returns the raw frame
        """
        return cv2.cvtColor(
            np.array(self.sct.grab(self.bounding_box)), cv2.COLOR_RGB2BGR
        )

    def view(self):
        """
        Shows the current frame
        """
        it = 0
        while True:
            it += 1
            # print(it)
            cur_frame = self.get_raw_frame()
            cv2.imshow("frame", cur_frame)
            cv2.imshow(
                "processed", cv2.resize(self.process_screen(cur_frame), (512, 192))
            )
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    viewer = GameViewer()
    viewer.view()
