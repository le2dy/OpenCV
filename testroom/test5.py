import cv2
import numpy as np

src = cv2.imread('../Images/paper2.png')

data = src.reshape(-1, 3).astype(np.float32)
crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, .001)

_, label, center = cv2.kmeans(data, 5, None, crit, 10, cv2.KMEANS_RANDOM_CENTERS)

center = center.astype(np.uint8)
dst = center[label].reshape(src.shape)

gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

# k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# gray = cv2.erode(dst, k)
# gray = cv2.dilate(dst, k)

_, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('gray', gray)
cv2.imshow('dst', thresh)
cv2.waitKey()
cv2.destroyAllWindows()
