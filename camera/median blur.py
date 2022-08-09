import cv2
import numpy as np

src = cv2.imread("../Images/noise.png")

blur = cv2.medianBlur(src, 5)

merge = np.hstack((src, blur))
cv2.imshow('merge', merge)
cv2.waitKey(0)
cv2.destoryAllWindows()
