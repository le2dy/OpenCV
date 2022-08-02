import cv2
import numpy as np

src = cv2.imread('../Images/cup.png')
rows, cols = src.shape[:2]


dx, dy = 100, 50

matrix = np.float32([[1, 0, dx],[0, 1, dy]])

dst = cv2.warpAffine(src,matrix, (cols + dx, rows + dy))
dst2 = cv2.warpAffine(src,matrix, (cols + dx, rows + dy), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255, 0, 0))
dst3 = cv2.warpAffine(src,matrix, (cols + dx, rows + dy), None, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

cv2.imshow('dst', dst)
cv2.imshow('dst2', dst2)
cv2.imshow('dst3', dst3)

cv2.waitKey(0)
cv2.destroyAllWindows()
