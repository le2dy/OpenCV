import cv2
import numpy as np

alpha = .5

src1 = cv2.imread('Images/downtown.png')
src2 = cv2.imread('Images/road.png')

src1 = cv2.resize(src1, (1920, 1260))

blended = src1 * alpha + src2 * (1 - alpha)
blended = blended.astype(np.uint8)

cv2.imshow('blended', blended)

dst = cv2.addWeighted(src1, alpha, src2, 1 - alpha, 0)

dst = np.concatenate((blended, dst), axis=1)
dst = cv2.resize(dst, (1920, 1080))

cv2.imshow('cv2 blended', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
