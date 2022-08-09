import cv2
import numpy as np

src = cv2.imread('../Images/noise.png')

k1 = np.array([[1, 2, 1],
               [2, 4, 2],
               [1, 2, 1]]) * (1 / 16)
blur1 = cv2.filter2D(src, -1, k1)

k2 = cv2.getGaussianKernel(3, 0)
blur2 = cv2.filter2D(src, -1, k2 * k2.T)

blur3 = cv2.GaussianBlur(src, (3, 3), 0)

print("k1", k1)
print("k2", k2 * k2.T)
merge = np.hstack((src, blur1,blur2,blur3))
merge = cv2.resize(merge, (1920, 640))
cv2.imshow('merge', merge)
cv2.waitKey()
cv2.destroyAllWindows()
