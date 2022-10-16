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


TEST = "TrackMania Nations Forever (TMInterface 1.1.1)"


LEFT, TOP, WIDTH, HEIGHT = getWindowGeometry(TEST)
print(TOP, LEFT, WIDTH, HEIGHT)


class GameViewer:
    def __init__(self) -> None:

        self.sct = mss()

    def process_screen(self, screenshot):
        baw = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        baw = cv2.Canny(baw, threshold1=100, threshold2=300)
        element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))

        baw = cv2.dilate(baw, element, iterations=3)
        baw = cv2.GaussianBlur(baw, (3, 3), 0)

        cut = baw[96:224, :]

        final = cv2.resize(cut, (128, 128))

        for i in range(1, 21):
            direction = np.pi * i / 20
            collison = self.find_end(direction, final)
            final = cv2.circle(final, collison, 1, (255, 255, 255), 1)
            # final = cv2.line(final, (64, 127), collison, (255, 255, 255), 2)

        return final

    def raycasting(self, frame, N_ranges=16):
        pass

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

        return int(cur_x), int(cur_y)

    def ROICrop(self, screenshot, x, y, w, h):
        return screenshot[y : y + h, x : x + w]

    def capture_and_save(self, save_path="./images", save=False):
        it = 0
        while True:
            self.bounding_box = getWindowGeometry(TEST)
            it += 1
            sct_img = np.array(
                cv2.resize(
                    cv2.cvtColor(
                        np.array(self.sct.grab(self.bounding_box)), cv2.COLOR_RGBA2RGB
                    ),
                    (256, 256),
                )
            )
            processed = self.process_screen(sct_img)
            cv2.imshow("screen", cv2.resize(sct_img, (512, 512)))
            cv2.imshow("processed", cv2.resize(processed, (512, 512)))

            if it % 100 == 0 and save:
                print(it)
                path = os.path.join(save_path, "screen_" + str(it) + ".jpg")
                cv2.imwrite(path, sct_img)
                print("saved")

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                cv2.destroyAllWindows(sct_img)

                break

    def get_frame(
        self,
    ):
        self.bounding_box = getWindowGeometry(TEST)
        sct_img = cv2.resize(
            cv2.cvtColor(
                np.array(self.sct.grab(self.bounding_box)), cv2.COLOR_RGBA2RGB
            ),
            (128, 128),
        )
        sct_img = self.process_screen(sct_img)
        res = np.array(sct_img)
        return res


if __name__ == "__main__":
    viewer = GameViewer()
    viewer.capture_and_save(save=False)
