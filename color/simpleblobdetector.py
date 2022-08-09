import cv2
import numpy as np

src = cv2.imread('../Images/downtown.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

detector = cv2.SimpleBlobDetector_create()

keypoints = detector.detect(gray)
src = cv2.drawKeypoints(src, keypoints, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
