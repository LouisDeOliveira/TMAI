import cv2
import numpy as np
import win32.win32gui as wind32
from mss import mss
from tmai.env.utils.constants import GAME_WINDOW_NAME


def getWindowGeometry(name: str) -> tuple:
    """
    Get the geometry of a window.
    """
    hwnd = wind32.FindWindow(None, name)
    left, top, right, bottom = wind32.GetWindowRect(hwnd)

    return left + 10, top + 40, right - 10, bottom - 10


class GameViewer:
    def __init__(self, n_rays: int = 16) -> None:
        self.window_name = GAME_WINDOW_NAME
        self.sct = mss()
        self.n_rays = n_rays

    @property
    def bounding_box(self):
        return getWindowGeometry(self.window_name)

    def process_screen(self, screenshot: np.ndarray) -> np.ndarray:
        baw = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        baw = cv2.threshold(baw, 32, 255, cv2.THRESH_BINARY)[1]
        baw = cv2.Canny(baw, threshold1=100, threshold2=300)
        element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
        baw = cv2.dilate(baw, element, iterations=3)
        baw = cv2.GaussianBlur(baw, (3, 3), 0)
        baw = cv2.threshold(baw, 1, 255, cv2.THRESH_BINARY)[1]

        baw = cv2.resize(baw, (128, 128))
        height = len(baw)
        cut = baw[height // 2 : height // 2 + 32, :]
        return cut

    def show_rays(self, frame):
        """
        Shows the rays of the frame
        """
        rays, _ = self.get_rays(frame, keep_horizontal=False)
        ref_point = (len(frame[0]) // 2, len(frame) - 1)
        for ray in rays:
            cv2.line(frame, ref_point, tuple(ray), (255, 0, 0), 1)

        return frame

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

    def _scaling_func(self, angle):
        return (1 + 3 * np.sin(angle)) / 4

    def get_distance(self, point, ref_size, ref_point=(64, 127), angle=0):
        return (
            self._scaling_func(angle)
            * np.linalg.norm(np.array(point) - np.array(ref_point), 2)
            / ref_size
        )

    def get_rays(self, frame, keep_horizontal=True):
        """
        Returns the rays of the frame
        """
        rays = []
        angles = []
        iterator = range(self.n_rays) if keep_horizontal else range(1, self.n_rays - 1)
        for i in iterator:
            angle = i * np.pi / (self.n_rays - 1)
            rays.append(self.find_end(angle, frame))
            angles.append(angle)
        return rays, angles

    def get_obs(self):
        processed_img = self.get_frame()
        ref_size = np.hypot(processed_img.shape[0], processed_img.shape[1]) / 2
        rays, angles = self.get_rays(processed_img)
        ref_point = (len(processed_img[0]) // 2, len(processed_img) - 1)
        distances = [
            self.get_distance(ray, ref_size, ref_point, angle)
            for ray, angle in zip(rays, angles)
        ]

        return np.array(distances).astype(np.float32)

    def get_frame(
        self,
        size=(256, 256),
    ) -> np.ndarray:
        """
        Pulls a frame from the game and processes it

        Args:
            size (tuple, optional): size to resize the screenshot to.
            Defaults to (256, 256).

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
            cur_frame = self.get_raw_frame()
            processed_frame = self.process_screen(cur_frame)
            raytrace = self.show_rays(processed_frame)

            cv2.imshow("frame", cur_frame)
            cv2.imshow(
                "processed",
                cv2.resize(
                    raytrace,
                    (512, 192),
                ),
            )
            if it % 20 == 0:
                obs = self.get_obs()
                print(min(obs))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    viewer = GameViewer()
    viewer.view()
