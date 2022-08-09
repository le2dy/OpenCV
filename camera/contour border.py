import cv2
import numpy as np

src = cv2.imread('../Images/skull.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

ctr = contours[0]

x, y, w, h = cv2.boundingRect(ctr)
cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 0), 3)

rect = cv2.minAreaRect(ctr)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(src, [box], -1, (0, 255, 0), 1)

(x, y), radius = cv2.minEnclosingCircle(ctr)
cv2.circle(src, (int(x), int(y)), int(radius), (255, 0, 0), 2)

ret, tri = cv2.minEnclosingTriangle(ctr)
cv2.polylines(src, [np.int32(tri)], True, (255, 0, 255), 2)

ellipse = cv2.fitEllipse(ctr)
cv2.ellipse(src, ellipse, (0, 255, 255), 3)

[vx, vy, x, y] = cv2.fitLine(ctr, cv2.DIST_L2, 0, .01, .01)
cols, rows = src.shape[:2]
cv2.line(src, (0, int(0 - x * (vy / vx) + y)), (cols - 1, int((cols - x) * (vy / vx) + y)), (0, 0, 255), 2)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
