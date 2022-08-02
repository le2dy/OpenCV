import cv2
import numpy as np

src = cv2.imread('Images/test.png')

data = src.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.001)
retval, bestLabels, centers = cv2.kmeans(data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = centers.astype(np.uint8)
dst = centers[bestLabels].reshape(src.shape)

cv2.imwrite('Images/saveImage.jpg', dst)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows
