import cv2
import numpy as np

src = cv2.imread('../Images/paper.png')
src = cv2.resize(src, None, fx=.5, fy=.5)

dst = cv2.pyrMeanShiftFiltering(src, 50, 60, None, 1)
dst = cv2.bilateralFilter(dst, 5, 75, 75)

gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
edge = cv2.Canny(gray, 50, 200)

# contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(dst, contours, -1, (0, 0, 255), 3)

corner = cv2.goodFeaturesToTrack(edge, 4, 0.1, 100)
corner = np.int32(corner)

for c in corner:
    x, y = c[0]
    cv2.circle(dst, (x, y), 5, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('src', src)
cv2.imshow('thresh', edge)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
