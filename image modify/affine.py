import cv2
import numpy as np
import matplotlib.pyplot as plt

name = "../Images/skull.png"

src = cv2.imread(name)
src = cv2.resize(src, (200, 200))

rows, cols = src.shape[:2]

pts1 = np.float32([[100, 50], [200, 50], [100, 200]])
pts2 = np.float32([[80, 70], [210, 60], [250, 120]])

cv2.circle(src, (100, 50), 5, (255, 0), -1)
cv2.circle(src, (200, 50), 5, (0, 255, 0), -1)
cv2.circle(src, (100, 200), 5, (0, 0, 255), -1)

matrix = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(src, matrix, (int(cols * 1.5), rows))

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
