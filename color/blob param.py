import cv2
import numpy as np

src = cv2.imread('../Images/downtown.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 10
params.maxThreshold = 240
params.thresholdStep = 5
params.filterByArea = True
params.minArea = 200

params.filterByColor = False
params.filterByConvexity = False
params.filterByColor = False
params.filterByColor = False

detector = cv2.SimpleBlobDetector_create(params)

keypoints= detector.detect(gray)

draw = cv2.drawKeypoints(src, keypoints, None, None, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('draw',draw)
cv2.waitKey(0)
cv2.destroyAllWindows()
