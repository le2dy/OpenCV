import cv2
import numpy as np

src = cv2.imread("../Images/skull.png")
src2 = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

print(thresh)

contour, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, \
                                      cv2.CHAIN_APPROX_NONE)
contour2, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print('Figure count: ', len(contour), len(contour2))

cv2.drawContours(src, contour, -1, (0, 255, 0), 4)
cv2.drawContours(src, contour2, -1, (0, 0, 255), 4)

for i in contour:
    for j in i:
        cv2.circle(src, tuple(j[0]), 1, (255, 0, 0), -1)

for i in contour2:
    for j in i:
        cv2.circle(src2, tuple(j[0]), 1, (0, 255, 0), -1)

cv2.imshow('none', src)
cv2.imshow('simple', src2)
cv2.waitKey()
cv2.destroyAllWindows()
