import cv2
import numpy as np

name = '../Images/skull.png'

src = cv2.imread(name)
src = cv2.resize(src, (500, 500))

rows, cols = src.shape[:2]

pts1 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
pts2 = np.float32([[100, 50], [10, rows - 50], [cols - 100, 50], [cols - 10, rows - 50]])

cv2.circle(src, (0, 0), 10, (255, 0, 0), -1)
cv2.circle(src, (0, rows), 10, (0, 255, 0), -1)
cv2.circle(src, (cols, 0), 10, (0, 0, 255), -1)
cv2.circle(src, (cols, rows), 10, (0, 255, 255), -1)

matrix = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(src, matrix, (cols, rows))

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
