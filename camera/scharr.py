import cv2
import numpy as np

# src = cv2.imread('../Images/beer.png')
src = cv2.imread('../Images/Document/img .png')

gx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
gy = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

edge_gx = cv2.filter2D(src, -1, gx)
edge_gy = cv2.filter2D(src, -1, gy)

scharr_x = cv2.Scharr(src, -1, 1, 0)
scharr_y = cv2.Scharr(src, -1, 0, 1)

merge1 = np.hstack((src, edge_gx, edge_gy, edge_gx + edge_gy))
merge2 = np.hstack((src, scharr_x, scharr_y, scharr_x + scharr_y))
merge = np.vstack((merge1, merge2))

merge = cv2.resize(merge, (1920, 1080))

cv2.imshow('merge', merge)
cv2.waitKey(0)
cv2.destroyAllWindows()
