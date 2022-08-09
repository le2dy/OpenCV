import cv2
import numpy as np

src = cv2.imread('../Images/paper.png')
src = cv2.resize(src, None, fx=.5, fy=.5)

blur1 = cv2.GaussianBlur(src, (5, 5), 0)
blur2 = cv2.bilateralFilter(src, 5, 75, 75)

gray = cv2.cvtColor(blur2, cv2.COLOR_BGR2GRAY)

gx_k = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
gy_k = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

edge_gx = cv2.filter2D(gray, -1, gx_k)
edge_gy = cv2.filter2D(gray, -1, gy_k)

scharrx = cv2.Scharr(gray, -1, 1, 0)
scharry = cv2.Scharr(gray, -1, 0, 1)

scharr = scharrx + scharry

cv2.imshow('src', src)
cv2.imshow('dst', scharr)
cv2.waitKey()
cv2.destroyAllWindows()
