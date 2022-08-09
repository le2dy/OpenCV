import cv2
import numpy as np

src = cv2.imread('../Images/city.png', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('../Images/home.jpg', cv2.IMREAD_GRAYSCALE)

src2 = cv2.resize(src2, (1920, 1080))

k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

opening = cv2.morphologyEx(src, cv2.MORPH_OPEN, k)
closing = cv2.morphologyEx(src, cv2.MORPH_CLOSE, k)

merge1 = np.hstack((src, src))
merge2 = np.hstack((opening, closing))
merge = np.vstack((merge1, merge2))
merge = cv2.resize(merge, (1920, 1080))

cv2.imshow('merge', merge)
cv2.waitKey()
cv2.destroyAllWindows()
