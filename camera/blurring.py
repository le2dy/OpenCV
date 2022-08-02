import cv2
import numpy as np

blur = 10
src = cv2.imread("../Images/beer.png")
kernel = np.ones((blur, blur)) / blur ** 2

blurred = cv2.filter2D(src, -1, kernel)

dst = np.concatenate((src, blurred), axis=1)

dst = cv2.resize(dst, (1080, 640))

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
