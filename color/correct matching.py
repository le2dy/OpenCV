import cv2
import numpy as np

src = cv2.imread('../Images/hat.png')
src2 = cv2.imread('../Images/hats.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

detector = cv2.ORB_create()

kp1, desc1 = detector.detectAndCompute(gray, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(desc1, desc2)

matches = sorted(matches, key=lambda x: x.distance)

min_dist, max_dist = matches[0].distance, matches[-1].distance

ratio = .2

correct_thresh = (max_dist - min_dist) * ratio + min_dist
correct_matches = [m for m in matches if m.distance < correct_thresh]
print('matches: %d/%d, min: %.2f, max: %.2f, thresh: %.2f' % (
    len(correct_matches), len(matches), min_dist, max_dist, correct_thresh))

dst = cv2.drawMatches(src, kp1, src2, kp2, correct_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
dst = cv2.resize(dst, (1080, 640))

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
