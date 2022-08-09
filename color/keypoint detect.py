import cv2
import numpy as np

src = cv2.imread('../Images/skull.png')
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

gftt = cv2.GFTTDetector_create()
keypoints = gftt.detect(gray, None)

draw = cv2.drawKeypoints(src, keypoints, None)

cv2.imshow("GFTT", draw)
cv2.waitKey()
cv2.destroyAllWindows()
