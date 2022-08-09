import cv2
import numpy as np

src = cv2.imread('../Images/skull.png')

gx_kernel = np.array([[-1, 1]])
gy_kernel = np.array([[-1], [1]])

gx_edge = cv2.filter2D(src, -1, gx_kernel)
gy_edge = cv2.filter2D(src, -1, gy_kernel)
merge = cv2.add(gy_edge, gx_edge)


dst = np.hstack((src, gx_edge, gy_edge, merge))

dst = cv2.resize(dst, (1920, 1080))

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
