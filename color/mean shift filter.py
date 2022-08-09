import cv2
import numpy as np

src = cv2.imread("../Images/apple.png")


def onChange(x):
    sp = cv2.getTrackbarPos('sp', 'src')
    sr = cv2.getTrackbarPos('sr', 'src')
    lv = cv2.getTrackbarPos('lv', 'src')

    mean = cv2.pyrMeanShiftFiltering(src, sp, sr, None, lv)

    dst = np.hstack((src, mean))
    dst = cv2.resize(dst, (1080, 640))

    cv2.imshow('src', dst)


dst = np.hstack((src, src))
dst = cv2.resize(dst, (1080, 640))

cv2.imshow('src', dst)
cv2.createTrackbar('sp', 'src', 0, 100, onChange)
cv2.createTrackbar('sr', 'src', 0, 100, onChange)
cv2.createTrackbar('lv', 'src', 0, 5, onChange)
cv2.waitKey()
cv2.destroyAllWindows()
