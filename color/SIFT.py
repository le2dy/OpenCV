import cv2
import numpy as np

src = cv2.imread('../Images/road.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

keypoints, descriptor = sift.detectAndCompute(gray, None)
print('keypoint: ', len(keypoints), 'descrpitor: ', descriptor.shape)
print(descriptor)

draw = cv2.drawKeypoints(src, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('draw', draw)
cv2.waitKey()
cv2.destroyAllWindows()
