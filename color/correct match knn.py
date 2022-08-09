import cv2
import numpy as np

src = cv2.imread('../Images/hat.png')
src2 = cv2.imread('../Images/hats.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

detector = cv2.ORB_create()

kp1, desc1 = detector.detectAndCompute(gray, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)
matches = matcher.knnMatch(desc1, desc2, 2)

ratio = .75
good_matches = [fst for fst, snd in matches if fst.distance < snd.distance * ratio]

print('matches: %d / %d' % (len(good_matches), len(matches)))

dst = cv2.drawMatches(src, kp1, src2, kp2, good_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
dst = cv2.resize(dst, (1080, 640))

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
