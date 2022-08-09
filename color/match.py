import cv2
import numpy as np

src = cv2.imread('../Images/face.png')
src2 = cv2.imread('../Images/skull2.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

detector = cv2.xfeatures2d.SIFT_create()

kp1, desc1 = detector.detectAndCompute(gray, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = matcher.match(desc1, desc2)

res = cv2.drawMatches(src, kp1, src2, kp2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
res = cv2.resize(res, (1080, 640))

cv2.imshow('matcher with SIFT', res)
cv2.waitKey()
cv2.destroyAllWindows()
