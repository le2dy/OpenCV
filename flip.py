import cv2
import numpy as np

src = cv2.imread('Images/cup.png', cv2.IMREAD_COLOR)
dst = cv2.flip(src, -1)

dst = np.concatenate((src, dst), axis=1)

dst = cv2.resize(dst, (1920, 1080))

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
