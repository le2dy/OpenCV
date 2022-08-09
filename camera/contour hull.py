import cv2
import numpy as np

src = cv2.imread('../Images/skull.png')
src2 = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour = contours[0]
cv2.drawContours(src, [contour], -1, (0, 255, 0), 1)

hull = cv2.convexHull(contour)
cv2.drawContours(src2, [hull], -1, (0, 255, 0), 1)
print(cv2.isContourConvex(contour), cv2.isContourConvex(hull))

hull2 = cv2.convexHull(contour, returnPoints=False)
defects = cv2.convexityDefects(contour, hull2)

for i in range(defects.shape[0]):
    start, end, farthest, distance = defects[i, 0]
    farthest = tuple(contour[farthest][0])
    dist = distance/256.0
    if dist > 1:
        cv2.circle(src2, farthest, 3, (0, 0, 255), -1)

cv2.imshow('src', src)
cv2.imshow('src2', src2)
cv2.waitKey()
cv2.destroyAllWindows()
