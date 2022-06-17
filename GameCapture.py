import numpy as np
import cv2
from mss import mss
import os
import win32.win32gui as wind32



def getWindowGeometry(name:str)->tuple:
    """
    Get the geometry of a window.
    """
    hwnd = wind32.FindWindow(None, name)
    left, top, right, bottom = wind32.GetWindowRect(hwnd)
 
    return  left+10, top+40, right-10, bottom-10

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

        
        return baw

    def ROICrop(self, screenshot, x, y, w, h):
        return screenshot[y:y+h, x:x+w]

    def capture_and_save(self, save_path = "./images", save = True):
        it = 0
        while True:
            self.bounding_box = getWindowGeometry(TEST)
            it+=1
            sct_img = cv2.cvtColor(np.array(self.sct.grab(self.bounding_box)), cv2.COLOR_RGBA2RGB)
            processed = self.process_screen(sct_img)
            cv2.imshow('screen', sct_img)
            cv2.imshow('processed', processed)
           
            if it%100 == 0 and save:
                print(it)
                path = os.path.join(save_path, "screen_" + str(it) + ".jpg")
                cv2.imwrite(path, sct_img)
                print("saved")

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows(sct_img)
                
                break

    
if __name__ == "__main__":
    viewer = GameViewer()
    viewer.capture_and_save(save=False)