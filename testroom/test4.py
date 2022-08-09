import cv2
import numpy as np

src = cv2.imread('../Images/downtown.png')
src = cv2.resize(src, None, fx=.5, fy = .5)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

dst = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
dst = (dst / (dst.max() - dst.min()) * 255).astype(np.uint8)

skeleton = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('skeleton', skeleton)
cv2.waitKey()
cv2.destroyAllWindows()
