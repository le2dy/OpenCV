import cv2
import numpy as np

src = cv2.imread('../Images/field.png')
height, width = src.shape[:2]

small = np.float32([[0.5, 0, 0], [0, 0.5, 0]])

big = np.float32([[2, 0, 0], [0, 2, 0]])

dst = cv2.warpAffine(src, small, (int(height * 0.5), int(width * 0.5)))
dst2 = cv2.warpAffine(src, big, (int(height * 2), int(width * 2)))

dst3 = cv2.warpAffine(src, small, (int(height * 0.5), int(width * 0.5)), None, cv2.INTER_AREA)
dst4 = cv2.warpAffine(src, big, (int(height * 2), int(width * 2)), None, cv2.INTER_CUBIC)

cv2.imshow('origin', src)
cv2.imshow('small', dst)
cv2.imshow('big', dst2)
cv2.imshow('small AREA', dst3)
cv2.imshow('big CUBIC', dst4)
cv2.waitKey(0)
cv2.destroyAllWindows()
