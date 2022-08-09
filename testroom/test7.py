import cv2
import numpy as np


def click(event, x, y, flag, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('a')


cv2.namedWindow('control')
src = cv2.imread('../Images/downtown.png')
src2 = cv2.imread('../Images/face2.png')
src3 = cv2.imread('../Images/skull.png')
src4 = cv2.imread('../Images/apple.png')
src5 = cv2.imread('../Images/hat.png')

src = cv2.resize(src, (400, 400))
src2 = cv2.resize(src2, (100, 100))
src3 = cv2.resize(src3, (100, 100))
src4 = cv2.resize(src4, (100, 100))
src5 = cv2.resize(src5, (100, 100))

hs = np.hstack((src2, src3, src4, src5))
vs = np.vstack((src, hs))

cv2.imshow('control', vs)
cv2.setMouseCallback('control', click)
cv2.waitKey(0)
cv2.destroyAllWindows()
