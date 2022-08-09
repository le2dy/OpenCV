import cv2
import numpy as np

src = cv2.imread("../Images/city.png")
src2 = cv2.imread("../Images/downtown.png")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

detector = cv2.ORB_create()

kp, desc = detector.detectAndCompute(gray, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(desc, desc2)

matches = sorted(matches, key=lambda x: x.distance)

res1 = cv2.drawMatches(src, kp, src2, kp2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

src_pts = np.float32([kp[m.queryIdx].pt for m in matches])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
h, w = src.shape[:2]

pts = np.float32([[[0, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w - 1, 0]]])
dst = cv2.perspectiveTransform(pts, matrix)

src2 = cv2.polylines(src2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

matchesMask = mask.ravel().tolist()
res2 = cv2.drawMatches(src, kp, src2, kp2, matches, None, matchesMask=matchesMask, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

accuracy = float(mask.sum()) / mask.size
print('accuracy: %d / %d(%.2f)' % (mask.sum(), mask.size, accuracy))
res1 = cv2.resize(res1, (1080, 640))
res2 = cv2.resize(res2, (1080, 640))

cv2.imshow('match all', res1)
cv2.imshow('match inlier', res2)
cv2.waitKey()
cv2.destroyAllWindows()
