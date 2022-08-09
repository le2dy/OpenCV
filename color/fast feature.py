import cv2
import numpy as np

src = cv2.imread('../Images/downtown.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

fast = cv2.FastFeatureDetector_create(50)
keypoints = fast.detect(gray, None)

src = cv2.drawKeypoints(src, keypoints, None)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
