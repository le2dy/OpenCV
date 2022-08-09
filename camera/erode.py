import cv2
import numpy as np

src = cv2.imread('../Images/field.png')

k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
erosion = cv2.erode(src, k)

dst = np.hstack((src, erosion))
dst = cv2.resize(dst, (1920, 1080))

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
