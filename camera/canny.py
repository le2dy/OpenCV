import cv2
import numpy as np

src = cv2.imread('../Images/skull.png')

canny = cv2.Canny(src, 100, 200)

cv2.imshow('srx', src)
cv2.imshow('canny', canny)
cv2.waitKey()
cv2.destroyAllWindows()
