import cv2
import numpy as np

src = cv2.imread('../Images/road.png')

small = cv2.pyrDown(src)
big = cv2.pyrUp(src)

cv2.imshow('big', big)
cv2.imshow('src', src)
cv2.imshow('small', small)
cv2.waitKey(0)
cv2.destroyAllWindows()
