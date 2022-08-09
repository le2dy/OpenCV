import cv2
import numpy as np

src = cv2.imread('../Images/beer.png')

gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

edge_gx = cv2.filter2D(src, -1, gx)
edge_gy = cv2.filter2D(src, -1, gy)

sobel_x = cv2.Sobel(src, -1, 1, 0, ksize=3)
sobel_y = cv2.Sobel(src, -1, 0, 1, ksize=3)

dst = np.hstack((src, edge_gy, edge_gx, edge_gy + edge_gx))
dst2 = np.hstack((src, sobel_x, sobel_y, sobel_x + sobel_y))

merge = np.vstack((dst, dst2))

merge = cv2.resize(merge, (1920, 1080))
cv2.imshow('dst', merge)
cv2.waitKey()
cv2.destroyAllWindows()
