import cv2
import numpy as np

src = cv2.imread('../Images/city.png')

k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
gradient = cv2.morphologyEx(src, cv2.MORPH_GRADIENT, k)

merge = np.hstack((src, gradient))
merge = cv2.resize(merge, (1920, 1080))

cv2.imshow('merge', merge)
cv2.waitKey()
cv2.destroyAllWindows()
