import cv2
import numpy as np

src = cv2.imread('../Images/skull.png')
src = cv2.resize(src, (200, 200))

rows, cols = src.shape[:2]

d45 = 45.0 * np.pi / 180
d90 = 90.0 * np.pi / 180

m45 = np.float32([[np.cos(d45), -np.sin(d45), rows // 2], [np.sin(d45), np.cos(d45), -cols//4]])
m90 = np.float32([[np.cos(d90), -np.sin(d90), rows], [np.sin(d90), np.cos(d90), 0]])

r45 = cv2.warpAffine(src, m45, (cols, rows))
r90 = cv2.warpAffine(src, m90, (rows, cols))

cv2.imshow('origin', src)
cv2.imshow('45', r45)
cv2.imshow('90', r90)
cv2.waitKey(0)
cv2.destroyAllWindows()
