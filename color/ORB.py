import cv2
import numpy as np

src = cv2.imread('../Images/downtown.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

keypoints, descriptor = orb.detectAndCompute(gray, None)
print(descriptor.shape, descriptor)

draw = cv2.drawKeypoints(src, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('draw', draw)
cv2.waitKey()
cv2.destroyAllWindows()
