import cv2
import numpy as np

src = cv2.imread("Images/apple.png")

height, width, channel = src.shape
matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
dst = cv2.warpAffine(src, matrix, (width, height))

dst = np.concatenate((src, dst), axis=1)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
