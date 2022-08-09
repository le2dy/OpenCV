import cv2
import numpy as np

src = cv2.imread('../Images/downtown.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create(1000, 3, extended=True, upright=True)

keypoints, descriptor = surf.detectAndCompute(gray, None)
print(descriptor.shape, descriptor)

draw = cv2.drawKeypoints(src, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('draw', draw)
cv2.waitKey()
cv2.destroyAllWindows()

# 지원안함
