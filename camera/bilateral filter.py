import cv2
import numpy as np

src = cv2.imread('../Images/beer.png')

blur1 = cv2.GaussianBlur(src, (5, 5,), 0)

blur2 = cv2.bilateralFilter(src, 5, 75, 75)

dst = np.hstack((src, blur1, blur2))

dst = cv2.resize(dst, (1920, 1080))

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
