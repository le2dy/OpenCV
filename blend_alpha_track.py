import cv2
import numpy as np

src1 = cv2.imread('Images/downtown.png')
src2 = cv2.imread('Images/road.png')

win_name = 'dst'


def onChange(value):
    global src1, src2
    src1 = cv2.resize(src1, (1920, 1260))

    alpha = value / 100
    dst = cv2.addWeighted(src2, alpha, src1, 1 - alpha, 0)
    dst = cv2.resize(dst, (1080, 640))
    cv2.imshow(win_name, dst)


src1 = cv2.resize(src1, (1080, 640))
cv2.imshow(win_name, src1)
cv2.createTrackbar('track', win_name, 0, 100, onChange)

cv2.waitKey(0)
cv2.destroyAllWindows()
