import cv2
import numpy as np

src = cv2.imread('Images/tree.png')
bgr = cv2.imread('Images/tree.png', cv2.IMREAD_COLOR)
bgra = cv2.imread('Images/tree.png', cv2.IMREAD_UNCHANGED)

dst = np.concatenate((src, bgr), axis=1)

dst = cv2.resize(dst, (1920, 540))
bgra = cv2.resize(bgra, (960, 540))
a = cv2.resize(bgra[:, :, 3], (960, 540))

cv2.imshow('dst', dst)
cv2.imshow('bgra', bgra)
cv2.imshow('a', a)
cv2.waitKey(0)
cv2.destroyAllWindows()
