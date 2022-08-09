import cv2
import numpy as np

src = cv2.imread('../Images/hat.png')
src2 = cv2.imread('../Images/hats.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

detector = cv2.xfeatures2d.SIFT_create()

kp1, desc1 = detector.detectAndCompute(gray, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(check=50)

matcher = cv2.FlannBasedMatcher(index_params, search_params)

matches = matcher.match(desc1, desc2)

dst = cv2.drawMatches(src, kp1, src2, kp2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
dst = cv2.resize(dst, (1080, 640))

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
