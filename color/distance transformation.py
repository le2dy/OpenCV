import cv2
import numpy as np

src = cv2.imread('../Images/skull.png', cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY_INV)

dst = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
dst = (dst / (dst.max() - dst.min()) * 255).astype(np.uint8)

skeleton = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)

dst = np.hstack((src, dst, skeleton))
dst = cv2.resize(dst, (1920, 1080))

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
