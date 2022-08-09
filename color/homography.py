import cv2
import numpy as np

src = cv2.imread("../Images/101_ObjectCategories/accordion/image_0001.jpg")
src2 = cv2.imread("../Images/101_ObjectCategories/accordion/image_0002.jpg")
gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

detector = cv2.ORB_create()

kp, desc = detector.detectAndCompute(src, None)
kp2, desc2 = detector.detectAndCompute(src2, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)
matches = matcher.knnMatch(desc, desc2, 2)

ratio = .75

good_matches = [fst for fst, snd in matches if fst.distance < snd.distance * ratio]
print('good matches: %d / %d' % (len(good_matches), len(matches)))

src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

matrix, mask = cv2.findHomography(src_pts, dst_pts)

h, w = src.shape[:2]
pts = np.float32([[[0, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w - 1, 0]]])

dst = cv2.perspectiveTransform(pts, matrix)

src2 = cv2.polylines(src2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
res = cv2.drawMatches(src, kp, src2, kp2, good_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
res = cv2.resize(res, (1080, 640))

cv2.imshow('res', res)
cv2.waitKey()
cv2.destroyAllWindows()
