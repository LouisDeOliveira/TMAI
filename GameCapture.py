from pickletools import read_uint1
from xml.dom.minidom import Element
import numpy as np
import cv2
from mss import mss
from collections import deque
bounding_box = {'top': 40, 'left': 0, 'width': 400, 'height': 304}

sct = mss()

def process_screen(screenshot):
    baw = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    baw = cv2.Canny(baw, threshold1=100, threshold2=300)
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))
    
    baw = cv2.dilate(baw, element, iterations=3)
    baw = cv2.GaussianBlur(baw, (3, 3), 0)

    
    return baw

def ROICrop(screenshot, x, y, w, h):
    return screenshot[y:y+h, x:x+w]

stacked = deque([np.zeros((204,400)),np.zeros((204,400)),np.zeros((204,400))], maxlen=5)
while True:
    sct_img = np.array(sct.grab(bounding_box))
    cut_img = ROICrop(sct_img, 0, 100, 400, 304)
    processed_img = process_screen(cut_img)
    
    # stacked.append(processed_img)
    # stacked_img = cv2.threshold((np.sum([img for img in stacked], axis=0)/3).astype(np.uint8),0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cv2.imshow('screen', sct_img)
    cv2.imshow('cut', cut_img)
    cv2.imshow('processed', processed_img)
    cv2.imshow('stacked', stacked_img)
    cv2.imwrite('./screenshot.png', sct_img)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows(sct_img)
        
        break

    

