import cv2
import numpy as np

src = cv2.imread('Images/apple.png')
src2 = cv2.imread('Images/cup.png')

x, y = src.shape[:2]

src2 = cv2.resize(src2, (y, x))

mask = np.full_like(src, 255)

h, w = src2.shape[:2]
center = (w//2, h//2)

normal = cv2.seamlessClone(src, src2, mask, center, cv2.NORMAL_CLONE)
mixed = cv2.seamlessClone(src, src2, mask, center, cv2.MIXED_CLONE)

cv2.imshow('normal', normal)
cv2.imshow('mixed', mixed)
cv2.waitKey(0)
cv2.destroyAllWindows()
