import cv2
import numpy as np

src = cv2.imread('../Images/apple.png')
src2 = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour = contours[22]

epsilon = .05 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)

cv2.drawContours(src, [contour], -1, (0, 255, 0), 3)
cv2.drawContours(src2, [approx], -1, (0, 255, 0), 3)

cv2.imshow('src', src)
cv2.imshow('src2', src2)
cv2.waitKey(0)
cv2.destroyAllWindows()
